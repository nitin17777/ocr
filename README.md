# OCR LoRA Correction — Evaluation Harness

A local, reproducible Python framework to evaluate and benchmark the `deep1104/ocr-lora-large` LoRA adapter on `Qwen/Qwen2.5-1.5B-Instruct` for synthetic OCR error correction.

---

## 1. Setup & Environment

### Prerequisites
- Python 3.10+ (tested on Python 3.13 64-bit on Windows 11)
- NVIDIA GPU (tested on RTX 2050 4GB with CUDA 12.9) or CPU fallback
- Read-only Hugging Face access token for the private adapter `deep1104/ocr-lora-large`

### Installation
1. Create and activate a virtual environment:
   ```powershell
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   ```
2. Install PyTorch with CUDA support:
   ```powershell
   pip install torch --index-url https://download.pytorch.org/whl/cu124
   ```
3. Install pinned requirements:
   ```powershell
   pip install -r requirements.txt
   ```
4. Configure credentials safely:
   Copy `.env.example` to `.env` and fill in your Hugging Face token:
   ```powershell
   cp .env.example .env
   # Edit .env and set HF_TOKEN=your_token_here
   ```
   > **Security Note**: Never commit `.env` or paste tokens into chat. `.gitignore` is pre-configured to ignore all `.env` files and data/output directories.

---

## 2. Checkpoint Verification

To inspect an existing evaluation checkpoint (e.g. `./data/ocr_eval_checkpoint.jsonl`) without running any model inference:

```powershell
.\venv\Scripts\python scripts/verify_checkpoint.py data/ocr_eval_checkpoint.jsonl
```

This verifies:
- Contiguity of indices and total record count (14,549 expected)
- Identification of bad/unparsed lines, duplicate indices, or gaps
- Baseline exact-match accuracy overall, and split by math sentences vs WikiText prose

---

## 3. Correcting Sentences (Inference)

You can correct any single sentence or enter an interactive terminal session:

### Direct CLI Correction:
```powershell
.\venv\Scripts\python -m ocr_lora.infer "Mu1tiply 76S by 17 to get l3005."
# Output: Multiply 765 by 17 to get 13005.
```

### Interactive Live Session:
```powershell
.\venv\Scripts\python -m ocr_lora.infer --interactive
```

### Use in Python Code:
```python
from ocr_lora.model import load_model_and_adapter
from ocr_lora.infer import correct_text_batch

model, tokenizer = load_model_and_adapter()
predictions = correct_text_batch(model, tokenizer, ["2O2 − 6 = 196.", "Ful1 of the true..."])
print(predictions)
```

---

## 4. Running & Resuming Evaluations

The evaluation harness implements safe atomic writes with `os.fsync` after each batch, never mutates the original checkpoint in `data/`, and writes to `outputs/ocr_eval_checkpoint.jsonl`.

### Smoke Test (Quick Validation)
Run a fast 50-row validation and preview 5 prediction samples:
```powershell
.\venv\Scripts\python -m ocr_lora.evaluate --limit 50 --samples 5
```

### Full / Resumed Evaluation
To evaluate all remaining unevaluated rows:
```powershell
.\venv\Scripts\python -m ocr_lora.evaluate --batch-size 8
```
- **Resumability**: If interrupted, simply re-run the same command. It automatically scans `outputs/ocr_eval_checkpoint.jsonl`, reads already completed record indices, and resumes from the first missing row.

---

## 4. Generating the Error Analysis Report

To generate the comprehensive failure analysis and confusion matrices:
```powershell
.\venv\Scripts\python -m ocr_lora.report
```
The report is rendered to `outputs/report.md`.

---

## 5. Running Tests

Execute unit tests covering prompt formatting, corruption rules, and resumable state logic:
```powershell
.\venv\Scripts\pytest tests/
```

---

## 6. Retraining (Optional)

Retrain the LoRA adapter using `trl==0.12.0` `SFTTrainer` with 4-bit nf4 quantization on an NVIDIA GPU:

```powershell
.\venv\Scripts\python train.py --epochs 3 --batch-size 4 --grad-accum 2 --lr 2e-4
```

- **Output**: Trained adapter weights are saved strictly locally to `models/ocr-lora-retrained/`.
- **Safety**: Adapters are **never** pushed to Hugging Face Hub unless `--push-to-hub` and `--repo-id <name>` are explicitly specified.

---

## 7. How to Share Safely

Before sharing this repository:
1. Ensure `.env` is absent (`.gitignore` protects it).
2. Check that no model checkpoints or downloaded weights from `models/` or raw benchmark datasets in `data/` or `outputs/` are included.
3. Keep code completely decoupled from hardcoded credentials.

---

## 8. Web Application (UI & API)

A local web application providing an interactive correction interface, batched file processing with live Server-Sent Events (SSE), an evaluation analytics dashboard, and a LoRA fine-tuning subprocess manager with GPU safety guards.

### Running the Web Server
Launch the server via the batch script:
```cmd
run_web.bat
```
Or with Uvicorn:
```powershell
.\venv\Scripts\python -m uvicorn webapp.main:app --host 127.0.0.1 --port 8000
```
- **Web UI**: http://127.0.0.1:8000
- **Interactive Swagger API Docs**: http://127.0.0.1:8000/docs
- **Health API**: http://127.0.0.1:8000/api/health

### Web Application Features

1. **Interactive Correction (Tab 1 — Correct)**:
   - Real-time character diff visualization using `difflib.SequenceMatcher` opcodes (green for insertions/replacements, red for deletions/replacements).
   - Hotkey support (`Ctrl+Enter` / `Cmd+Enter`), character counters, and sample pre-fill buttons.
   - Dynamic PEFT adapter switching with in-place adapter swapping and lazy loading.
   - VRAM usage and generation latency badges.

2. **Batch File Processing (Tab 2 — Batch)**:
   - Supports `.txt` (one sentence per line), `.csv` (automatic header detection, UTF-8 with BOM), and `.jsonl` uploads.
   - Interactive column/field selector for CSV and JSONL files.
   - Real-time SSE progress streaming: processed rows, percentage, rows/sec, and dynamic ETA calculation.
   - Crash-resilient resumable processing (skips already evaluated rows in `results.jsonl`).
   - 10-row preview table with character-level diffs and one-click export to CSV (Excel compatible) or JSONL.

3. **Evaluation Analytics (Tab 3 — Analytics)**:
   - Read-only inspection of the 14,549-row evaluation benchmark (`data/ocr_eval_checkpoint.jsonl`).
   - In-memory mtime/size caching serving sub-5ms responses.
   - Executive metrics breakdown: Total Rows (14,549), Exact-Match Accuracy (90.70%), Math Accuracy (98.41%), and Text Accuracy (51.48%).
   - Character confusion matrix heatmap rendered as an interactive CSS grid with intensity scaling, hover tooltips, and click-to-filter capability.
   - Failures explorer with debounced keyword search, domain filters, pagination, and side-by-side ground truth vs. model prediction diffs.

4. **Fine-Tuning Runner & Monitor (Tab 4 — Training)**:
   - Subprocess execution of `train.py` using virtual environment Python with process group isolation.
   - 4GB VRAM safety preflight: unloads the inference model from GPU memory and locks inference endpoints (returns `503 {"reason": "training_in_progress"}`).
   - Parameter bounds enforcement: epochs (1–10), batch size (1–4), learning rate ($10^{-6}$ to $10^{-3}$), and regex-validated target adapter directory names.
   - Live console terminal window streaming stdout/stderr via SSE with auto-scroll toggles and ANSI code stripping.
   - Process tree termination via `taskkill /F /T` with zero orphaned `python.exe` processes.
   - Past runs history viewer and automatic adapter catalog refresh upon completion.

### Rebuilding Tailwind CSS (Optional)
The pre-compiled CSS bundle is already committed at `webapp/static/css/tailwind.css`. No Node.js or npm is required at runtime.

If you modify `index.html` or `app.js` and wish to rebuild the stylesheet:
1. Download the standalone Tailwind CLI for Windows (`tailwindcss-windows-x64.exe`) from GitHub:
   https://github.com/tailwindlabs/tailwindcss/releases/latest/download/tailwindcss-windows-x64.exe
2. Place the executable as `tailwindcss.exe` in the repository root.
3. Run the build script:
   ```cmd
   webapp\static\build_css.bat
   ```


