"""Retraining script for OCR LoRA model using TRL SFTTrainer.

Strictly adheres to confirmed notebook specifications:
- Base model: Qwen/Qwen2.5-1.5B-Instruct
- Prompt: "### Instruction:\nFix the OCR errors: {input}\n\n### Response:\n{expected}"
- LoRA: r=16, alpha=32, dropout=0.05, target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj']
- Training: 3 epochs, batch_size=4, grad_accum=2, lr=2e-4, cosine, warmup 0.1, max_seq_len 256
- 4-bit nf4 quantization (with FP16 fallback)
- Saves adapter locally; never pushes to Hub without explicit user confirmation.
"""

from typing import Optional, Dict, Any
import os
import sys
import argparse
from pathlib import Path
import json

import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer

# Import distributed tensor patch for Windows PyTorch
try:
    import torch.distributed.tensor
except ImportError:
    pass

from ocr_lora.config import (
    BASE_MODEL_ID,
    PROMPT_TEMPLATE,
    TRAIN_LORA_R,
    TRAIN_LORA_ALPHA,
    TRAIN_LORA_DROPOUT,
    TRAIN_TARGET_MODULES,
    TRAIN_EPOCHS,
    TRAIN_BATCH_SIZE,
    TRAIN_GRAD_ACCUM,
    TRAIN_LR,
    TRAIN_LR_SCHEDULER,
    TRAIN_WARMUP_RATIO,
    TRAIN_MAX_SEQ_LEN,
    CHECKPOINT_INPUT_PATH,
    MODELS_DIR,
    get_hf_token,
)


def format_training_example(example: Dict[str, str]) -> Dict[str, str]:
    """Format single input/expected pair into full prompt + response string."""
    prompt = PROMPT_TEMPLATE.format(input=example["input"])
    full_text = prompt + example["expected"]
    return {"text": full_text}


def load_training_dataset(data_path: Path) -> Dataset:
    """Load JSONL dataset and prepare for SFTTrainer."""
    if not data_path.exists():
        raise FileNotFoundError(f"Training dataset not found at {data_path}")

    records = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                records.append({
                    "input": row["input"],
                    "expected": row["expected"],
                })
            except Exception:
                continue

    raw_ds = Dataset.from_list(records)
    formatted_ds = raw_ds.map(format_training_example)
    return formatted_ds


def train(
    data_path: Path = CHECKPOINT_INPUT_PATH,
    output_dir: Path = MODELS_DIR / "ocr-lora-retrained",
    epochs: int = TRAIN_EPOCHS,
    batch_size: int = TRAIN_BATCH_SIZE,
    grad_accum: int = TRAIN_GRAD_ACCUM,
    lr: float = TRAIN_LR,
    use_4bit: bool = True,
    push_to_hub: bool = False,
    hub_repo_id: Optional[str] = None,
) -> None:
    """Execute SFTTrainer pipeline on NVIDIA GPU."""
    if not torch.cuda.is_available():
        raise RuntimeError("Retraining strictly requires an NVIDIA GPU (CUDA). CUDA is not available on this device.")

    token = get_hf_token()
    print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
    print(f"Base model: {BASE_MODEL_ID}")
    print(f"Output adapter directory: {output_dir}")

    # Quantization config
    bnb_config = None
    if use_4bit:
        try:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
            print("Configured 4-bit nf4 quantization.")
        except Exception as e:
            print(f"Warning: Failed to configure BitsAndBytesConfig ({e}). Falling back to FP16.")
            bnb_config = None

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL_ID,
        token=token,
        trust_remote_code=True,
    )
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Load base model
    model_kwargs: Dict[str, Any] = {
        "token": token,
        "trust_remote_code": True,
        "device_map": "auto",
    }
    if bnb_config is not None:
        model_kwargs["quantization_config"] = bnb_config
    else:
        model_kwargs["torch_dtype"] = torch.float16

    print("Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_ID, **model_kwargs)

    if bnb_config is not None:
        model = prepare_model_for_kbit_training(model)

    # Configure LoRA
    lora_config = LoraConfig(
        r=TRAIN_LORA_R,
        lora_alpha=TRAIN_LORA_ALPHA,
        lora_dropout=TRAIN_LORA_DROPOUT,
        target_modules=TRAIN_TARGET_MODULES,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # Prepare dataset
    print(f"Loading dataset from {data_path}...")
    dataset = load_training_dataset(data_path)
    print(f"Dataset size: {len(dataset)} examples")

    output_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        lr_scheduler_type=TRAIN_LR_SCHEDULER,
        warmup_ratio=TRAIN_WARMUP_RATIO,
        fp16=True,
        logging_steps=10,
        save_strategy="epoch",
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=lora_config,
        dataset_text_field="text",
        max_seq_length=TRAIN_MAX_SEQ_LEN,
        tokenizer=tokenizer,
        args=training_args,
    )

    print("\n=== Commencing Training ===")
    trainer.train()

    print(f"\nSaving fine-tuned adapter to {output_dir}...")
    trainer.model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    print(f"Adapter successfully saved locally to {output_dir}")

    # Never push to Hub without explicit confirmation
    if push_to_hub and hub_repo_id:
        print(f"Pushing adapter to Hub repository '{hub_repo_id}'...")
        trainer.model.push_to_hub(hub_repo_id, token=token)
        tokenizer.push_to_hub(hub_repo_id, token=token)
        print("Push to Hub complete.")
    else:
        print("Preserved adapter strictly locally (no Hub upload).")


def main():
    parser = argparse.ArgumentParser(description="OCR LoRA Retraining Script")
    parser.add_argument("--data", type=str, default=str(CHECKPOINT_INPUT_PATH), help="Path to JSONL training data")
    parser.add_argument("--output", type=str, default=str(MODELS_DIR / "ocr-lora-retrained"), help="Output adapter directory")
    parser.add_argument("--epochs", type=int, default=TRAIN_EPOCHS, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=TRAIN_BATCH_SIZE, help="Batch size")
    parser.add_argument("--grad-accum", type=int, default=TRAIN_GRAD_ACCUM, help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=TRAIN_LR, help="Learning rate")
    parser.add_argument("--no-4bit", action="store_true", help="Disable 4-bit nf4 quantization (use FP16)")
    parser.add_argument("--push-to-hub", action="store_true", help="Push adapter to Hugging Face Hub (requires --repo-id)")
    parser.add_argument("--repo-id", type=str, default=None, help="Target Hugging Face repository ID")
    args = parser.parse_args()

    train(
        data_path=Path(args.data),
        output_dir=Path(args.output),
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        lr=args.lr,
        use_4bit=not args.no_4bit,
        push_to_hub=args.push_to_hub,
        hub_repo_id=args.repo_id,
    )


if __name__ == "__main__":
    main()
