# -*- coding: utf-8 -*-
"""WasteClassifier.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1qMxyoVngHiV6E2ePu54o5PiJFr5Ifhku

# Waste Classifier

This model has been trained using Fastai. 

**Fastai** is a deep learning library which provides high-level components that can quickly and easily provide state-of-the-art results in standard deep learning domains.

The aim of this project is to build a model for waste classification that identifies among the different classes:
- cardboard
- compost
- glass
- metal
- paper
- plastic
- trash

This machine learning model will help people to improve their decision when classifying trash. 

The model has been trained on a set of images which were obtained using **Bing** searcher using the api HTTP.
Those images has been manually cleaned, removing the ones that were not usefull or where in the wrong category.

## Transfer learning 
**ResNet50** is a pre-trained convolutional neural network(CNN) that has 50 layers. It has been already trained with images from the ImageNet database. It classifies 1000 object from very broad categories, such as pencil or animals. The input size of the network is 224x224. This network can be reused to train other model.

Thus, Resnet50 has been used to train the model for the 7 classes.

# Install
Install fastai library
"""

pip uninstall fastai
pip uninstall fastai2
pip install fastai==2.5.3

pip install Streamlit

from fastai.vision.all import *
from google.colab import files
import cv2
import matplotlib.pyplot as plt
import zipfile
import io
import os
from PIL import Image

"""# Load model
Let's first download the git repository and load the *model*
"""

!git clone https://github.com/rootstrap/fastai-waste-classifier

"""Set fastai-waste-classifier as the current directory"""

# Commented out IPython magic to ensure Python compatibility.
# %cd fastai-waste-classifier
import utils

"""Load the trained model with fastai and use it to classify some images. """

learn_loaded = load_learner('result-resnet34.pkl')

"""# Load files
Load your files to be classified

## Case 1: zip file 
"""

uploaded = files.upload()

filename = next(iter(uploaded))
data = zipfile.ZipFile(filename, 'r')
data.extractall() 
filename = filename.split('.')[0]

"""Remove unwanted files , check that all files in the directory are images """

! rm $filename/'.DS_Store'

def check_folder(folder_path):
    for f in os.listdir(folder_path):
      file_path = os.path.join(folder_path, f)
      print('** Path: {}  **'.format(file_path), end="\r", flush=True)
      im = Image.open(file_path)
      rgb_im = im.convert('RGB')

check_folder(filename)

"""Get the predictions for all the images """

predictions = utils.get_predictions(learn_loaded, filename)

"""Plot the images and the obtained predictions"""

rows = round(len(predictions)/5)
_, axs = plt.subplots(rows, 5, figsize=(20, 20))
axs = axs.flatten()
for img, ax, p in zip(os.listdir(filename), axs, predictions):
    image=Image.open(f'{filename}/{img}')
    ax.set_title(p[1])
    ax.imshow(image)
    ax.set_xticks([])
    ax.set_yticks([])
  
plt.show()

"""## Case 2: Upload 1 file to be classified"""

def predict_image():
  uploaded = files.upload()
  filename = next(iter(uploaded))
  img = Image.open(f'{filename}')
  plt.figure()
  plt.xticks([])
  plt.yticks([])
  plt.imshow(img) 
  plt.show() 

  # Predict value for the uploaded file
  prediction = learn_loaded.predict(filename)
  num = prediction[1].numpy().tolist()
  print(f'Classified as {prediction[0]}', f'Class number {num}', f' with probability {prediction[2].numpy()[num]}')
  return prediction[0]

"""Calling to predict_image method, you can upload the image that you want to classify and it will print the result"""

predict_image()

predict_image()

predict_image()
