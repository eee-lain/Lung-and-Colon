import streamlit as st
import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
from PIL import Image

# Load your trained model
model = tf.keras.models.load_model('Keras/trained_lung_colon_cnn.keras')

# Define the class mappings
cls = {
    0: 'Colon adenocarcinoma',
    1: 'Colon Normal',
    2: 'Lung adenocarcinoma',
    3: 'Lung Normal',
    4: 'Lung squamous cell carcinoma'
}

# Streamlit app
st.title('Lung and Colon Cancer Detection')

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    img = Image.open(uploaded_file)
    st.image(img, caption='Uploaded Image', use_column_width=True)
    st.write("")

    # Preprocess the image
    img = np.array(img)  # Convert to numpy array
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert color from BGR to RGB if needed
    img = cv2.resize(img, (128, 128))  # Resize the image to match the input size of the model
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Normalize the image

    # Make predictions
    predictions = model.predict(img_array)
    result_index = np.argmax(predictions)
    model_prediction = cls[result_index]

    # Display the result
    #st.write(f"Disease Name: {model_prediction}")

    # Display the image with the prediction as title
    plt.imshow(img)
    plt.title(f"Disease Name: {model_prediction}")
    plt.axis('off')
    st.pyplot(plt)
