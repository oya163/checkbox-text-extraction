---
title: Checkbox and Text Detector
emoji: 🏆
colorFrom: red
colorTo: yellow
sdk: gradio
sdk_version: 4.36.1
app_file: app.py
python_version: 3.11.2
pinned: false
---

# Checkbox Detection and Text Extraction

## Introduction

- This tool helps you to extract the checked text (The text next to the checked box).
- It is built on top of [Checkbox-Detection](https://github.com/LynnHaDo/Checkbox-Detection) by [LynnHaDo](https://github.com/LynnHaDo).
- It uses the **Euclidean Distance** to extract the checked text following the result from checkbox detection model and the OCR
![Checkbox Text Extraction](images/checked_text_extraction.png "Checkbox Text Extraction")

## Workflow
![Checkbox Detection and corresponding text extraction](images/checkbox_detection_workflow.png "Checkbox Detection and Text Extraction Workflow")

## Installation

    # Create virtual environment
    python3.11 -m venv .venv

    # Activate the virtual environment
    source .venv/bin/activate

    # Install the requirements from requirements.txt
    pip install -r requirements.txt

    # Install poppler-utils
    sudo apt-get install poppler-utils libpoppler-cpp-dev

## Demo

This is currently deployed in [HuggingFace Spaces](https://huggingface.co/spaces/oyashi163/checkbox_text_extraction)

## Note

Please use the sample image provided in `images` folder for testing purpose. 
You might want to modify the OCR parameters for your image to get more accurate later.

You should be able to find the `models/detector-model.pt` file here in [google drive](https://drive.google.com/file/d/1S1yo_l-BPpaPc6pWwdqeHI4gA0Xue2wg/view?usp=sharing). If not, please email me.

## References
- https://github.com/LynnHaDo/Checkbox-Detection
