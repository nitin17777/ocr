"""Inference module for batched OCR error correction.

Executes greedy decoding (do_sample=False) with left padding, max_length=256 truncation,
and max_new_tokens=100. Accurately extracts and strips only newly generated tokens.
"""

from typing import List
import torch
from transformers import PreTrainedTokenizer, PreTrainedModel

from ocr_lora.config import (
    PROMPT_TEMPLATE,
    EVAL_MAX_LENGTH,
    EVAL_MAX_NEW_TOKENS,
    EVAL_DO_SAMPLE,
)


def format_prompt(input_text: str) -> str:
    """Format single input text with exact confirmed prompt template."""
    return PROMPT_TEMPLATE.format(input=input_text)


def correct_text_batch(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    input_texts: List[str],
    max_length: int = EVAL_MAX_LENGTH,
    max_new_tokens: int = EVAL_MAX_NEW_TOKENS,
    do_sample: bool = EVAL_DO_SAMPLE,
) -> List[str]:
    """Correct OCR errors for a batch of input texts.
    
    Decodes only newly generated tokens and strips whitespace.
    """
    prompts = [format_prompt(text) for text in input_texts]

    # Tokenize with left padding and truncation
    encoded = tokenizer(
        prompts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )

    device = next(model.parameters()).device
    encoded = {k: v.to(device) for k, v in encoded.items()}

    input_seq_len = encoded["input_ids"].shape[1]

    # Support both Qwen stop tokens: <|im_end|> (151645) and <|endoftext|> (151643)
    eos_token_ids = [151645, 151643]
    if hasattr(model, "generation_config") and model.generation_config.eos_token_id:
        cfg_eos = model.generation_config.eos_token_id
        if isinstance(cfg_eos, list):
            eos_token_ids = cfg_eos
        elif isinstance(cfg_eos, int):
            eos_token_ids = [cfg_eos]

    with torch.no_grad():
        generated_ids = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=eos_token_ids,
        )

    # Slice only new tokens
    new_tokens = generated_ids[:, input_seq_len:]
    decoded = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)

    return [d.strip() for d in decoded]


def main():
    """CLI and interactive entrypoint for OCR correction."""
    import argparse
    from ocr_lora.model import load_model_and_adapter
    from ocr_lora.config import DEFAULT_ADAPTER_ID, BASE_MODEL_ID

    parser = argparse.ArgumentParser(description="OCR LoRA Correction Inference")
    parser.add_argument("text", nargs="?", default=None, help="Input text with OCR errors to correct")
    parser.add_argument("--adapter", type=str, default=DEFAULT_ADAPTER_ID, help="Adapter ID or path")
    parser.add_argument("--base-model", type=str, default=BASE_MODEL_ID, help="Base model ID")
    parser.add_argument("--interactive", "-i", action="store_true", help="Launch interactive prompt")
    args = parser.parse_args()

    print(f"Loading model ({args.base_model}) with adapter ({args.adapter})...")
    model, tokenizer = load_model_and_adapter(
        base_model_id=args.base_model,
        adapter_id=args.adapter,
    )
    print("Model loaded successfully!\n")

    if args.interactive or not args.text:
        print("=== Interactive OCR Correction Mode ===")
        print("Type any corrupted sentence and press Enter (or type 'exit' to quit):\n")
        while True:
            try:
                user_input = input("OCR Input > ").strip()
                if not user_input or user_input.lower() in ("exit", "quit"):
                    break
                corrected = correct_text_batch(model, tokenizer, [user_input])[0]
                print(f"Corrected > {corrected}\n")
            except (KeyboardInterrupt, EOFError):
                break
        print("Exiting.")
    else:
        corrected = correct_text_batch(model, tokenizer, [args.text])[0]
        print(f"Input:     {args.text}")
        print(f"Corrected: {corrected}")


if __name__ == "__main__":
    main()
