# Talrn Text-to-Image Generator

AI-powered text-to-image generation using Stable Diffusion v1.5 with Streamlit UI.

## Features
- Text-to-image generation with adjustable parameters
- Automatic watermarking with AI origin indicator
- PNG + JPEG export formats
- Metadata storage (prompt, parameters, timestamp)
- Works on CPU and GPU
- Open-source and ethical AI

## Installation

```bash
git clone https://github.com/rasalageethanjali-ai/talrn-text-to-image.git
cd talrn-text-to-image
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage

```bash
streamlit run app.py
```

Open http://localhost:8501 in your browser.

## How to Use

1. Enter a detailed prompt
2. Adjust quality and creativity settings
3. Add negative prompts if needed
4. Click Generate and wait
5. Images saved in `outputs/` folder

## Performance

- CPU (8GB): 1-5 mins per image
- GPU (8GB VRAM): 10-30 secs per image

## Technical Stack

- Model: Stable Diffusion v1.5
- Framework: PyTorch + Diffusers
- UI: Streamlit
- Storage: Local filesystem + JSON

## Submitted for

Talrn ML Internship Task Assessment
