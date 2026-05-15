import streamlit as st
import tensorflow as tf
import numpy as np

# Tensrorflow Model Prediction
def model_prediction(test_image):
    # Load the pre-trained model
    model = tf.keras.models.load_model('trained_model.keras')

    # Preprocess the image
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])  # Convert single image to a batch. (1, 128, 128, 3)

    # Make prediction

    result_index = np.argmax(prediction)

    return result_index


# Streamlit App

#Sidebar
