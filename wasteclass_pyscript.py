# -*- coding: utf-8 -*-
"""
WasteClassifier.py
"""

# Install dependencies
!pip uninstall fastai
!pip uninstall fastai2
!pip install fastai==2.5.3
!pip install Streamlit

# Import libraries
from fastai.vision.all import *
from google.colab import files
import cv2
import matplotlib.pyplot as plt
import zipfile
import io
import os
from PIL import Image
import streamlit as st
import utils

# Load model
st.set_option('deprecation.showfileUploaderEncoding', False)
st.title("Waste Classifier")

@st.cache(allow_output_mutation=True)
def load_model():
    learn_loaded = load_learner('result-resnet34.pkl')
    return learn_loaded

model = load_model()

def predict_image(filename):
    # Predict value for the uploaded file
    prediction = model.predict(filename)
    num = prediction[1].numpy().tolist()
    st.write(f'Classified as {prediction[0]}')
    st.write(f'Class number {num}')
    st.write(f'With probability {prediction[2].numpy()[num]}')

# Load files
st.write("Upload image for classification")
file = st.file_uploader("", type=["jpg","jpeg","png"])
if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    predict_image(file)
