# Facial Expression Recognition using CNN + LSTM

This version is dataset-driven only. It does not use a webcam, camera stream, face detector, or real-time video processing.

The project trains on your existing folder dataset:

```text
face_expression_dataset/
├── train/
│   ├── angry/
│   ├── disgust/
│   ├── fear/
│   ├── happy/
│   ├── neutral/
│   ├── sad/
│   └── surprise/
└── test/
    ├── angry/
    ├── disgust/
    ├── fear/
    ├── happy/
    ├── neutral/
    ├── sad/
    └── surprise/
```

## What It Does

- Loads images from `face_expression_dataset/train` and `face_expression_dataset/test`
- Trains a CNN + LSTM classifier
- Uses CNN layers for spatial face-expression features
- Reshapes CNN feature maps into spatial sequences for a bidirectional LSTM
- Applies augmentation during training
- Uses capped class weights to help with class imbalance without overcorrecting
- Saves the trained model automatically
- Evaluates accuracy and saves a confusion matrix
- Provides a Streamlit UI for uploaded image prediction

## Emotions

- Angry
- Disgust
- Fear
- Happy
- Neutral
- Sad
- Surprise

## Project Structure

```text
emotion-recognition/
├── app/
├── dataset/
├── models/
├── notebooks/
├── app.py
├── preprocess.py
├── predict.py
├── train.py
├── requirements.txt
└── README.md
```

Your actual training data remains outside the project folder at:

```text
../face_expression_dataset
```

## Installation

Use the TensorFlow-compatible Python version already available on your machine. This project now targets TensorFlow `2.20` or `2.21`, which matches the versions your pip resolver can install.

```powershell
cd C:\Users\hp\face_expression_recognition\emotion-recognition
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Inspect Dataset

```powershell
python preprocess.py --dataset ..\face_expression_dataset
```

## Train

```powershell
python train.py --dataset ..\face_expression_dataset --epochs 35 --batch-size 64
```

For a faster first run:

```powershell
python train.py --dataset ..\face_expression_dataset --epochs 10 --batch-size 64
```

If overall accuracy is still low, try disabling class weights. This may improve top-line accuracy, though minority-class recall can drop:

```powershell
python train.py --dataset ..\face_expression_dataset --epochs 25 --batch-size 64 --no-class-weights
```

Generated files:

```text
models/best_cnn_lstm_expression_model.keras
models/cnn_lstm_expression_model.keras
models/confusion_matrix.png
models/training_curves.png
models/history.json
models/metadata.json
```

## Predict One Image

```powershell
python predict.py --image ..\face_expression_dataset\test\happy\PrivateTest_647018.jpg
```

Use any image path from your dataset.

## Run Streamlit App

```powershell
streamlit run app.py
```

The app lets you upload a face image and displays:

- Predicted emotion
- Confidence percentage
- Probability bars for every class

## GPU Notes

If TensorFlow detects a CUDA-capable GPU, `train.py` enables GPU memory growth automatically. If no GPU is available, it trains on CPU.

## Important Note About CNN + LSTM on Static Images

Your dataset contains static images, not videos. The code keeps the requested CNN + LSTM architecture by treating CNN feature-map patches as a sequence for the LSTM. This is better for image-folder datasets than repeating the same image across fake timesteps.
