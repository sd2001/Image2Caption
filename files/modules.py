import os
from pandas import DataFrame
from tensorflow.keras.preprocessing import image, sequence
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
import ssl
import plotly
import logging, os
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
from PIL import Image
from pickle import dump,load
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input,Dense,LSTM,Embedding,Dropout,add
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import Model,load_model
from nltk.translate.bleu_score import corpus_bleu
tf.get_logger().setLevel('INFO')
from numpy import argmax,array
import ssl
from os import listdir
import matplotlib.pyplot as plt

IMG_DIR="/home/sd2001/Desktop/Programming/Image Captioning/Flickr8k_Dataset/Flicker8k_Dataset"
TRAIN_IMG_NAME="/home/sd2001/Desktop/Programming/Image Captioning/Flickr8k_text/Flickr_8k.trainImages.txt"
TEST_IMG_NAME="/home/sd2001/Desktop/Programming/Image Captioning/Flickr8k_text/Flickr_8k.testImages.txt"
VALID_IMG_NAME="/home/sd2001/Desktop/Programming/Image Captioning/Flickr8k_text/Flickr_8k.devImages.txt"
IMG_CAPTION="/home/sd2001/Desktop/Programming/Image Captioning/Flickr8k_text/Flickr8k.token.txt"
LEMM_CAPTION="/home/sd2001/Desktop/Programming/Image Captioning/Flickr8k_text/Flickr8k.lemma.token.txt"
