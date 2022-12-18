import pandas as pd
import json
import cv2 as cv
from sklearn.neighbors import NearestNeighbors
import streamlit as st
import PIL
from joblib import load
import numpy as np


def vectorize(model,img):
    sift = cv.SIFT_create()
    _,ides = sift.detectAndCompute(cv.cvtColor(img,cv.COLOR_BGR2GRAY),None)
    classes = model.predict(ides)
    hist,_ = np.histogram(classes,1024,[0,1024])
    return 1024 * hist / sum(hist)

@st.cache
def load_base():
    model = load('model.joblib')
    db = pd.read_csv('base.csv', delimiter=',')
    db['Vec'] = db['Vec'].apply(lambda x: json.loads(x))
    return model,db

model,db = load_base()
st.header("Similar Image Search")
uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg", "webp", "tiff"])
if uploaded_file is not None:
    image = PIL.Image.open(uploaded_file)
    st.success("This is your image:")
    st.image(image)
    number = st.number_input('Insert a number of neighbors',min_value=1,max_value=10,step=1)
    neighbors_model = NearestNeighbors(n_neighbors=number, metric='cosine')
    neighbors_model.fit(db['Vec'].tolist())
    
    st.success("This is what I found:")
    image = np.array(image)[...,::-1]
    
    neighbors = neighbors_model.kneighbors([vectorize(model,image)], number, return_distance=False)[0]
    col = st.columns(3)
    for i in range(len(neighbors)):
        with col[i%3]:
            st.image(PIL.Image.open(db['Path'][neighbors[i]]))
