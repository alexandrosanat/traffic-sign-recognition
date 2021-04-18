import os
import tensorflow as tf
from PIL import Image
import glob
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st


def intro():
    # Page Title
    st.title("""
     Traffic sign classifier
     """)

    # Page Intro
    st.write("""
     Welcome to the prototype Traffic Sign Classifier. 
     This app uses a LeNet-5 CNN trained using the German
     Traffic Sign Recognition Benchmark dataset available [here]
     (https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/published-archive.html
     ) to identify common traffic signs.
     """)


def sidebar():
    # Sidebars
    st.sidebar.header("About")
    st.sidebar.text("This is ...")


def prediction_text():
    st.write("The model is trained to recognise the below categories:")


def load_signal_categories():
    cwd = os.getcwd()
    categories_path = os.path.join(cwd, "data_included", "signal categories")
    image_list = []
    for filename in glob.glob(os.path.join(categories_path, "*.png")):
        im = Image.open(filename)
        image_list.append(im)

    return image_list


def main():

    # Page Intro
    intro()

    # Sidebars
    sidebar()

    # Create two columns for the model plots
    col1, col2 = st.beta_columns(2)

    # Plot model summary plots
    cwd = os.getcwd()
    training_plots_path = os.path.join(cwd, 'outputs')
    plots = list()

    for i, img_name in enumerate(os.listdir(training_plots_path)):
        img_path = os.path.join(training_plots_path, img_name)
        img = tf.keras.preprocessing.image.load_img(img_path)
        plots.append(img)

    # Training Accuracy Plot
    col1.image(plots[0], channels='RGB')

    # Training Loss Plot
    col2.image(plots[1], channels='RGB')

    # Text for included categories
    prediction_text()

    category_images = load_signal_categories()

    fig1, ax1 = plt.subplots(1, 15)
    fig2, ax2 = plt.subplots(1, 15)
    fig3, ax3 = plt.subplots(1, 13)
    fig1.tight_layout()
    fig2.tight_layout()
    fig3.tight_layout()

    for i in range(15):
        ax1[i].imshow(category_images[i])
        ax1[i].set_axis_off()

    for i in range(15):
        ax2[i].imshow(category_images[i+15])
        ax2[i].set_axis_off()

    for i in range(13):
        ax3[i].imshow(category_images[i+30])
        ax3[i].set_axis_off()

    st.pyplot(fig1)
    st.pyplot(fig2)
    st.pyplot(fig3)

    st.write("You can try the model by uploading your own photo of a traffic sign below:")

    st.write("NOTE: This model expects pictures of individual signs, so remove or"
             "crop out background items for best results.")

    uploaded_image = st.file_uploader(label="", type=["png", "jpg"])

    # Tensorflow GPU settings
    physical_devices = tf.config.list_physical_devices("GPU")
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

    # Load trained model
    cwd = os.getcwd()
    model_path = os.path.join(cwd, "trained_model/lenet_64x64")
    model = tf.keras.models.load_model(model_path)

    if uploaded_image is not None:

        # Create two columns for the predicted plots
        col3, col4 = st.beta_columns(2)

        # Predict image
        image = Image.open(uploaded_image)

        st.write("")
        col3.write("Classifying...")

        col3.image(image.resize((150, 150)), caption='', use_column_width=False)

        x = image.resize((64, 64)).convert("L")
        img = tf.keras.preprocessing.image.img_to_array(x)
        img = np.expand_dims(img, axis=0)

        prediction = np.argmax(model.predict(img, steps=1), axis=1)

        col4.write("The predicted image is:")
        col4.image(category_images[prediction[0]])


if __name__ == '__main__':
    main()

