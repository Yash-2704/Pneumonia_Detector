# Pneumonia Detector Web App

Simple Flask interface that loads the trained ResNet50 model (`pneumonia_resnet50_final.h5`) and predicts whether an uploaded chest X-ray shows signs of pneumonia.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

> **Note:** If you are on Apple Silicon and already use `tensorflow-macos`, replace the TensorFlow line inside `requirements.txt` with the appropriate package before installing.

## Running the app

```bash
python app.py
```

Open http://127.0.0.1:5000 in your browser and upload a chest X-ray image. The page will display whether pneumonia is detected along with the model confidence.

The preprocessing pipeline matches training: each image is enhanced with CLAHE via OpenCV, converted back to RGB, resized to 384×384, and normalized to `[0,1]` before feeding the ResNet50 model.

## Project structure

```
Pneumonia_Detector/
├── app.py                 # Flask server and prediction logic
├── pneumonia_resnet50_final.h5  # Provided trained model
├── requirements.txt       # Python dependencies
├── templates/
│   └── index.html         # Upload form and results view
└── static/
    └── styles.css         # Minimal styling for the page
```
# Pneumonia_Detector
