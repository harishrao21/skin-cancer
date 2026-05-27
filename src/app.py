from __future__ import division, print_function
import sys
import os
import glob
import re
from pathlib import Path
from io import BytesIO
import base64
import requests
import torch
import numpy as np
import random

# Import fast.ai Library
from fastai.vision.all import *

# Flask utils
from flask import Flask, redirect, url_for, render_template, request
from PIL import Image as PILImageLib
from torch import nn
from torchvision import transforms
from torchvision.models import densenet169

# Define a flask app
app = Flask(__name__)

NAME_OF_FILE = 'model_best' # Name of your exported file
PATH_TO_MODELS_DIR = Path('models') # by default just use /models in root dir
classes = ['Actinic keratoses', 'Basal cell carcinoma', 'Benign keratosis',
           'Dermatofibroma', 'Melanocytic nevi', 'Melanoma', 'Vascular lesions']

# Global variable for the model (lazy loading)
learn = None

def load_model():
    """Load the pre-trained model (lazy loading on first use)"""
    global learn
    if learn is not None:
        return learn

    try:
        print("Loading model...")
        model_path = PATH_TO_MODELS_DIR / f'{NAME_OF_FILE}.pkl'

        # Try .pkl first (fastai saved model)
        if model_path.exists():
            from fastai.learner import load_learner
            learn = load_learner(model_path, cpu=True)
            print("Model loaded successfully!")
            return learn

        # Try .pth file and rebuild the original DenseNet169 architecture.
        model_path_pth = PATH_TO_MODELS_DIR / f'{NAME_OF_FILE}.pth'
        if model_path_pth.exists():
            print(f"Found .pth file at {model_path_pth}")
            checkpoint = torch.load(model_path_pth, map_location='cpu', weights_only=False)
            state_dict = checkpoint.get('model') if isinstance(checkpoint, dict) else None
            if state_dict is None:
                print("Checkpoint did not contain model weights; using demo mode.")
                return None

            backbone = densenet169(weights=None)
            body = nn.Sequential(backbone.features)
            head = nn.Sequential(
                AdaptiveConcatPool2d(),
                nn.Flatten(),
                nn.BatchNorm1d(3328),
                nn.Dropout(0.25),
                nn.Linear(3328, 512),
                nn.ReLU(inplace=True),
                nn.BatchNorm1d(512),
                nn.Dropout(0.5),
                nn.Linear(512, len(classes)),
            )
            learn = nn.Sequential(body, head)
            learn.load_state_dict(state_dict)
            learn.eval()
            print("Model loaded successfully!")
            return learn

        print(f"Model file not found at {model_path} or {model_path_pth}")
        print("Using demo mode...")
        return None

    except Exception as e:
        print(f"Error loading model: {e}")
        print("Using demo mode without actual model predictions...")
        return None


class AdaptiveConcatPool2d(nn.Module):
    def __init__(self, size=1):
        super().__init__()
        self.ap = nn.AdaptiveAvgPool2d(size)
        self.mp = nn.AdaptiveMaxPool2d(size)

    def forward(self, x):
        return torch.cat([self.mp(x), self.ap(x)], 1)


IMAGE_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])


def _looks_like_image_bytes(data):
    """Best-effort validation that payload is likely an image."""
    if not data or len(data) < 12:
        return False

    # Common magic bytes for JPEG, PNG, GIF, BMP.
    signatures = (
        b"\xff\xd8\xff",        # JPEG
        b"\x89PNG\r\n\x1a\n", # PNG
        b"GIF87a",               # GIF
        b"GIF89a",               # GIF
        b"BM",                   # BMP
    )

    is_webp = data.startswith(b"RIFF") and data[8:12] == b"WEBP"
    return data.startswith(signatures) or is_webp

def encode(img_pil):
    """Encode PIL image to base64"""
    buff = BytesIO()
    img_pil.save(buff, format="JPEG")
    return base64.b64encode(buff.getvalue()).decode("utf-8")
	
def model_predict(img_bytes):
    """Predict skin cancer type from image bytes"""
    global learn
    img_data = None
    try:
        # Load model on first use
        if learn is None:
            learn = load_model()
        
        # Open image from bytes
        try:
            if not _looks_like_image_bytes(img_bytes):
                return render_template(
                    'result.html',
                    result={
                        "error": "The provided file/URL is not a valid image. Please use JPG, PNG, GIF, WEBP, or BMP.",
                        "image": "error_placeholder",
                    },
                )
            # Verify catches corrupted files before we do full decode.
            PILImageLib.open(BytesIO(img_bytes)).verify()
            img = PILImageLib.open(BytesIO(img_bytes)).convert('RGB')
            img_data = encode(img)
        except Exception as img_error:
            print(f"Image loading error: {img_error}")
            return render_template(
                'result.html',
                result={
                    "error": "Could not decode the provided image. Please upload a valid JPG, PNG, GIF, WEBP, or BMP file.",
                    "image": "error_placeholder",
                },
            )
        
        # If model loaded successfully, make prediction
        if learn is not None:
            try:
                img_tensor = IMAGE_TRANSFORM(img).unsqueeze(0)
                with torch.no_grad():
                    outputs = learn(img_tensor)
                    probs = torch.softmax(outputs[0], dim=0)

                pred_idx = int(torch.argmax(probs).item())
                pred = classes[pred_idx]
                pred_probs = sorted(
                    zip(classes, [f"{value * 100:.1f}%" for value in probs.tolist()]),
                    key=lambda p: float(p[1].rstrip('%')),
                    reverse=True,
                )

                result = {"class": pred, "probs": pred_probs, "image": img_data}
                return render_template('result.html', result=result)
            except Exception as pred_error:
                print(f"Prediction error: {pred_error}")
                return render_template('result.html', result={"error": f"Prediction error: {pred_error}", "image": img_data})
        else:
            # Demo mode: return random classification
            print("Running in demo mode (model not available)")
            
            # Generate demo predictions
            demo_probs = []
            random_values = [random.uniform(0, 100) for _ in range(len(classes))]
            total = sum(random_values)
            random_values = [v/total * 100 for v in random_values]  # Normalize to 100%
            
            demo_probs = sorted(
                zip(classes, [f"{v:.1f}%" for v in random_values]),
                key=lambda p: float(p[1].rstrip('%')),
                reverse=True
            )
            
            predicted_class = demo_probs[0][0]
            result = {"class": predicted_class, "probs": demo_probs, "image": img_data, "demo": True}
            return render_template('result.html', result=result)
        
        return render_template('result.html', result={"error": "Unexpected prediction flow", "image": img_data})
            
    except Exception as e:
        print(f"Error in model_predict: {e}")
        import traceback
        traceback.print_exc()
        return render_template('result.html', result={"error": str(e), "image": img_data if img_data else "error_placeholder"})
   

@app.route('/', methods=['GET', "POST"])
def index():
    # Main page
    return render_template('index.html')


@app.route('/upload', methods=["POST", "GET"])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        file_obj = request.files.get('file')
        if file_obj is None:
            return render_template('result.html', result={"error": "No file was uploaded"})

        if not file_obj.filename:
            return render_template('result.html', result={"error": "Please choose an image file before submitting"})

        if not (file_obj.mimetype or "").startswith("image/"):
            return render_template('result.html', result={"error": "Uploaded file is not an image. Please upload JPG, PNG, GIF, WEBP, or BMP."})

        img = file_obj.read()
        if img is not None:
            # Make prediction
            preds = model_predict(img)
            return preds
        return render_template('result.html', result={"error": "Uploaded file could not be read"})
    return 'OK'
	
@app.route("/classify-url", methods=["POST", "GET"])
def classify_url():
    if request.method == 'POST':
        url = request.form.get("url")
        if not url:
            return render_template('result.html', result={"error": "No image URL was provided"})

        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
        except requests.RequestException as request_error:
            return render_template('result.html', result={"error": f"Could not fetch image URL: {request_error}"})

        content_type = response.headers.get("Content-Type", "")
        if content_type and not content_type.lower().startswith("image/"):
            return render_template(
                'result.html',
                result={"error": f"The URL did not return an image (Content-Type: {content_type}). Please provide a direct image link."},
            )

        preds = model_predict(response.content)
        return preds
    return 'OK'
    

if __name__ == '__main__':
    port = os.environ.get('PORT', 8008)

    if "prepare" not in sys.argv:
        app.run(debug=False, host='0.0.0.0', port=port)
