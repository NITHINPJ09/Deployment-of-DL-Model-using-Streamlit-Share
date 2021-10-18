import streamlit as st
from PIL import Image
import numpy as np
import cv2

st.title("Image Classifier")
st.header("CIFAR-10 Image Classification")
uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")
    label_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    label_names.sort()
    net = cv2.dnn.readNetFromONNX('cifar_classifier.onnx')
    img = cv2.resize(np.array(image),(32,32))
    img = np.array([img]).astype('float64') / 255.0
    net.setInput(img)
    out = net.forward()
    index = np.argmax(out[0])
    label =  label_names[index].capitalize()
    st.write(label)