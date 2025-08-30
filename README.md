# Arabic ASR with Wav2Vec 2.0 (Common Voice 11.0)

A clean, reproducible demo for **Arabic Automatic Speech Recognition (ASR)** built on `facebook/wav2vec2-base` and Mozilla **Common Voice** v11.0 (Arabic).  
This repo contains a ready-to-run Jupyter notebook, quick evaluation guidance (WER), and minimal dependencies for fast setup.


![wave2vec2-base](image.png)

## Why this project?
- Showcases an end-to-end ASR workflow in Arabic: data loading, resampling to 16 kHz, quick exploration, and inference.
- Easy to extend into proper fine-tuning and evaluation.
- Great as a learning artifact or portfolio piece.

## Contents
- `arabic-asr-wav2vec2-demo.ipynb` — main notebook (outputs cleared).
- `requirements.txt` — minimal dependencies.
- `.gitignore` — ignores caches and large local folders.
- `images/wav2vec2_slide.png` — reference slide used in the README (optional).

## Quickstart
```bash
# 1) Create & activate a virtual env (optional but recommended)
python -m venv .venv
source .venv/bin/activate  # on Windows: .venv\Scripts\activate

# 2) Install deps
pip install --upgrade pip
pip install -r requirements.txt

# 3) Launch the notebook
jupyter lab  # or: jupyter notebook
```

## Dataset
- **Mozilla Common Voice v11.0 (Arabic)** — `mozilla-foundation/common_voice_11_0`  
  You can load it via 🤗 Datasets inside the notebook.


## Quick Evaluation (WER)
If you have reference transcripts, you can compute **WER** quickly:

```python
from jiwer import wer
reference = "هذا اختبار"
hypothesis = pred_str  # from the model output
print("WER:", wer(reference, hypothesis))
```

