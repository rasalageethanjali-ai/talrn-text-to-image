#!/usr/bin/env python3
"""
Talrn Text-to-Image Generator
AI-powered text-to-image generation using Stable Diffusion v1.5
"""

import os
import json
from datetime import datetime
from PIL import Image, ImageDraw
import torch
from diffusers import StableDiffusionPipeline
import streamlit as st

# Page config
st.set_page_config(page_title="Talrn AI Image Generator", layout="wide")
st.title("✨ Talrn Text-to-Image Generator")

# Create output folder
os.makedirs("outputs", exist_ok=True)

# Load model (cached)
@st.cache_resource
def load_model():
    st.write("📦 Loading model...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float32 if device == 'cpu' else torch.float16
    )
    pipe.to(device)
    return pipe, device

pipe, device = load_model()

# Sidebar settings
st.sidebar.header("⚙️ Generation Settings")
prompt = st.sidebar.text_area("🎨 Prompt:", value="a futuristic city at sunset", height=80)
negative_prompt = st.sidebar.text_area("❌ Negative Prompt:", value="blurry, low quality", height=60)
num_steps = st.sidebar.slider("Quality (steps):", 10, 50, 20)
guidance_scale = st.sidebar.slider("Creativity:", 1.0, 15.0, 7.5)

# Generate button
if st.sidebar.button("🎨 Generate Image", use_container_width=True):
    with st.spinner("Generating... (takes 1-5 mins on CPU)"):
        try:
            result = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=num_steps,
                guidance_scale=guidance_scale
            )
            image = result.images[0]
            
            # Watermark
            draw = ImageDraw.Draw(image)
            draw.text((10, 10), "AI-Generated | Talrn", fill="white")
            
            # Save
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            folder = f"outputs/{timestamp}"
            os.makedirs(folder, exist_ok=True)
            
            image.save(f"{folder}/image.png")
            image.save(f"{folder}/image.jpg")
            
            metadata = {
                "timestamp": timestamp,
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "steps": num_steps,
                "guidance_scale": guidance_scale,
                "device": device
            }
            with open(f"{folder}/metadata.json", "w") as f:
                json.dump(metadata, f, indent=2)
            
            st.success(f"✅ Image generated! Saved to {folder}/")
            st.image(image, caption="Generated Image")
            
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")

# Instructions
st.markdown("""
## 📋 How to Use
1. Enter a detailed prompt
2. Adjust settings
3. Click Generate and wait

## ⏱️ Speed
- **CPU**: 1-5 mins/image
- **GPU**: 10-30 secs/image

## 📁 Output
Generated images saved in `outputs/` with metadata.
""")
