[README.md](https://github.com/user-attachments/files/22060067/README.md)
# Arabic ASR with Wav2Vec 2.0 (Common Voice 11.0)

A clean, reproducible demo for **Arabic Automatic Speech Recognition (ASR)** built on `facebook/wav2vec2-base` and Mozilla **Common Voice** v11.0 (Arabic).  
This repo contains a ready-to-run Jupyter notebook, quick evaluation guidance (WER), and minimal dependencies for fast setup.

![Wav2Vec 2.0](images/wav2vec2_slide.png)

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

## Minimal Inference (pseudo-snippet)
Below is a short reference for running a quick forward pass (actual code is in the notebook).

```python
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import torch, torchaudio

model_id = "facebook/wav2vec2-base"
processor = Wav2Vec2Processor.from_pretrained(model_id)
model = Wav2Vec2ForCTC.from_pretrained(model_id)

# Load mono audio at 16 kHz
wav, sr = torchaudio.load("sample.wav")
if sr != 16000:
    wav = torchaudio.functional.resample(wav, sr, 16000)

inputs = processor(wav.squeeze().numpy(), sampling_rate=16000, return_tensors="pt", padding=True)
with torch.no_grad():
    logits = model(inputs.input_values).logits

pred_ids = torch.argmax(logits, dim=-1)
pred_str = processor.batch_decode(pred_ids)[0]
print(pred_str)
```

## Quick Evaluation (WER)
If you have reference transcripts, you can compute **WER** quickly:

```python
from jiwer import wer
reference = "هذا اختبار"
hypothesis = pred_str  # from the model output
print("WER:", wer(reference, hypothesis))
```

## Notes
- This is a **learning/demo** repository. For full fine-tuning, consider adding language-specific tokenizers and CTC training scripts.
- The included slide image comes from your assignment PDF to illustrate context only.
- Credits: your course handout and slides.


---
*Prepared for GitHub publication. Feel free to rename the repo to something like `arabic-asr-wav2vec2-demo` and add screenshots/GIFs of results later.*
