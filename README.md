
---
title: Virtual Try-On Diffusion IEEE Dressify
emoji: 👗
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: 5.6.0
app_file: app.py
pinned: false
short_description: Diffusion-based multi-modal virtual try-on pipeline demo.
tags:
  - virtual try-on
  - vton
  - clothing transfer
  - diffusion
  - img2img
  - txt2img
---

# Virtual Try-On Diffusion [VTON-D] by IEEE Dressify

Virtual Try-On Diffusion [VTON-D] by Dressify is a custom diffusion-based pipeline designed for fast and flexible multi-modal virtual try-on. This system enables tasks such as clothing transfer, avatar replacement, fashion image generation, and more by using reference images or text prompts to specify clothing, avatar, and background.

---

## Installation

### 1. Clone the Repository
```bash
git clone https://github.com/Louay0007/Virtual-Try-On-Dressify.git
cd Virtual-Try-On-Dressify
```

### 2. Create a Virtual Environment
```bash
python -m venv venv
```

### 3. Activate the Virtual Environment
- **Windows**:
  ```bash
  venv\Scripts\activate
  ```
- **macOS/Linux**:
  ```bash
  source venv/bin/activate
  ```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## Usage

### 1. Run the Application
```bash
python app.py
```

### 2. Access the Application
Open your browser and navigate to:
```
http://127.0.0.1:7860
```

---

## Features

- **Clothing Transfer**: Transfer garments between avatars with precision.
- **Avatar Replacement**: Easily switch between different avatars.
- **Fashion Image Generation**: Generate new fashion images based on reference images or text descriptions.
- **Multi-Modal Input**: Use text prompts or images for customization.

---

## Tags

- Virtual Try-On
- VTON
- Clothing Transfer
- Diffusion
- Image-to-Image (img2img)
- Text-to-Image (txt2img)

---

## Demos Version

- [🤗](https://huggingface.co/spaces/Louu007/ISSATM-VTO)
---

This project is a personal one 
