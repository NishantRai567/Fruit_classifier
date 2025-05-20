# import libraries
import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img,img_to_array
from PIL import Image

# load our model pipeline object

model_filename='C:/Users/Nisha/OneDrive/Desktop/DATA SCIENCE INFINITY/Deep Learning/Deep Learning/CNN/models/fruits_cnn_01.h5'
model=load_model(model_filename)

# add title and instructions
st.title("Fruit Classifier 🍎🍌🍊")
st.write("Upload an image of a fruit and the model will classify it.")
labels_list=['apple','avocado','banana','kiwi','lemon','orange']

uploaded_file = st.file_uploader("Choose a fruit image...", type=["jpg", "jpeg", "png"])

# preprocesss image

if uploaded_file is not None:
    # Open the image with PIL
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    # Resize and preprocess the image
    image = image.resize((128, 128))  # Match your model's input size
    image = img_to_array(image)
    image = image*(1./255.0)
    image = np.expand_dims(image, axis=0)

    # predict image
    
    class_probs=model.predict(image)
    predicted_class=np.argmax(class_probs)
    predicted_label=labels_list[predicted_class]
    predicted_prob=class_probs[0][predicted_class]
     
    # Show result
    st.write(f"### Predicted Fruit: **{predicted_label}** ({predicted_prob:.2f} confidence)")
