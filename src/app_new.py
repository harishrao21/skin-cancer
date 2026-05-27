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
from PIL import Image as PILImage

# Define a flask app
app = Flask(__name__)

NAME_OF_FILE = 'model_best'
PATH_TO_MODELS_DIR = Path('models')
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
            print(f"Attempting to load .pkl file from {model_path}")
            from fastai.learner import load_learner
            learn = load_learner(model_path, cpu=True)
            print("Model loaded successfully!")
            return learn
        
        # Try .pth file
        model_path_pth = PATH_TO_MODELS_DIR / f'{NAME_OF_FILE}.pth'
        if model_path_pth.exists():
            print(f"Found .pth file at {model_path_pth}")
            learn = torch.load(model_path_pth, map_location='cpu')
            print("Model loaded successfully!")
            return learn
        
        print(f"Model file not found at {model_path} or {model_path_pth}")
        print("Using demo mode...")
        return None
        
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Using demo mode without actual model predictions...")
        return None


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
            img = PILImage.create(BytesIO(img_bytes))
            img_data = encode(img)
        except Exception as img_error:
            print(f"Image loading error: {img_error}")
            return render_template('result.html', result={"error": f"Image loading error: {img_error}", "image": "error_placeholder"})
        
        # If model loaded successfully, make prediction
        if learn is not None:
            try:
                # Make prediction
                pred, pred_idx, outputs = learn.predict(img)
                
                # Calculate probabilities
                probs = torch.nn.functional.softmax(outputs, dim=0)
                formatted_outputs = ["{:.1f}%".format(value * 100) for value in probs]
                
                # Sort by probability
                pred_probs = sorted(
                    zip(classes, map(str, formatted_outputs)),
                    key=lambda p: float(p[1].rstrip('%')),
                    reverse=True
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
            random_values = [v/total * 100 for v in random_values]
            
            demo_probs = sorted(
                zip(classes, [f"{v:.1f}%" for v in random_values]),
                key=lambda p: float(p[1].rstrip('%')),
                reverse=True
            )
            
            predicted_class = demo_probs[0][0]
            result = {"class": predicted_class, "probs": demo_probs, "image": img_data, "demo": True}
            return render_template('result.html', result=result)
            
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
        img = request.files['file'].read()
        if img != None:
            # Make prediction
            preds = model_predict(img)
            return preds
    return 'OK'


@app.route("/classify-url", methods=["POST", "GET"])
def classify_url():
    if request.method == 'POST':
        url = request.form["url"]
        if url != None:
            response = requests.get(url)
            img = response.content
            preds = model_predict(img)
            return preds
    return 'OK'


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8008, use_reloader=False)
