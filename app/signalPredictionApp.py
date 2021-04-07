import streamlit as st
import os
import tensorflow as tf

# Page Title
st.write("""
## Sign prediction app
""")

# Page Title
st.write("""
This app uses a LeNet 5 model trained over x epochs using the German
Traffic Sign Recognition Benchmark dataset of available here:
https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/published-archive.html


The dataset contains x number of traffic sign images.""")

cwd = os.getcwd()
training_plots_path = os.path.join(cwd, '..', 'outputs')
plots = list()

for i, img_name in enumerate(os.listdir(training_plots_path)):
    img_path = os.path.join(training_plots_path, img_name)
    img = tf.keras.preprocessing.image.load_img(img_path)
    plots.append(img)

# Training Accuracy Plot
st.image(plots[0], channels='RGB')

# Training Loss Plot
st.image(plots[1], channels='RGB')

