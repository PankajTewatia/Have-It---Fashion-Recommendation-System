import streamlit as st
from streamlit_option_menu import option_menu
import os
from PIL import Image
import glob
import cv2 as cv
import pandas as pd
import numpy as np
import pickle
import tensorflow
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50,preprocess_input
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import norm

selected = option_menu(None, ["Home", "Trending", "Textual Search", "Reverse Image Search", "About"],
                       icons=['house', 'bar-chart', "search", "search", "people"],
                       default_index=0, orientation="horizontal")


if selected == "Home":
    st.title(f"Have It!")


if selected == "Trending":
    st.title(f"Trending Now")
    path = glob.glob("D:/trending/img/*.jpg")
    cv_img = []
    for img in path:
        n = cv.imread(img)
        n = cv.resize(n, (200, 200))
        cv_img.append(n)
    st.image(cv_img, channels="BGR")


if selected == "Textual Search":
    st.title(f"Select Your Item")
    df = pd.read_csv("D:/trending/styles.csv", error_bad_lines=False)
    df['image'] = df.apply(lambda row: str(row['id']) + ".jpg", axis=1)
    df = pd.DataFrame({'name': df['image'], 'type': df['articleType']})
    unique_types = df['type'].unique().tolist()
    total_class = len(unique_types)
    sel = st.selectbox("Select your item: ", unique_types)
    result = df[df["type"] == sel]

    result_path = 'D:/trending/fashion-dataset/images/'+result['name']

    cv_img2 = []
    for img in result_path:
        n = cv.imread(img)
        n = cv.resize(n, (350, 500))
        cv_img2.append(n)
    # st.write(cv_img2)
    st.image(cv_img2, channels="BGR")



if selected == "Reverse Image Search":
    st.title(f"Find Best Matches from our WARDROBE INVENTORY")
    feature_list = np.array(pickle.load(open('embeddings.pkl', 'rb')))
    filenames = pickle.load(open('filenames.pkl', 'rb'))

    model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    model.trainable = False

    model = tensorflow.keras.Sequential([
       model,
       GlobalMaxPooling2D()
    ])



    def save_uploaded_file(uploaded_file):
        try:
            with open(os.path.join('uploads', uploaded_file.name), 'wb') as f:
                f.write(uploaded_file.getbuffer())
            return 1
        except:
            return 0


    def feature_extraction(img_path, model):
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        expanded_img_array = np.expand_dims(img_array, axis=0)
        preprocessed_img = preprocess_input(expanded_img_array)
        result = model.predict(preprocessed_img).flatten()
        normalized_result = result / norm(result)

        return normalized_result


    def recommend(features, feature_list):
        neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
        neighbors.fit(feature_list)

        distances, indices = neighbors.kneighbors([features])

        return indices

    uploaded_file = st.file_uploader("Choose an image")
    st.image("D:\\trending\\fashion-dataset\images\\1165.jpg")
    if uploaded_file is not None:
        if save_uploaded_file(uploaded_file):
            # display the file
            display_image = Image.open(uploaded_file)
            st.image(display_image)
            # feature extract
            features = feature_extraction(os.path.join("uploads", uploaded_file.name), model)
            # st.text(features)
            # recommendention
            indices = recommend(features, feature_list)
            # show
            col1, col2, col3, col4, col5 = st.columns(5)

            with col1:
                # path = glob.glob("D:\\trending\\fashion-dataset\\" + filenames[indices[0][0]])
                # n = cv.imread(str(path))
                # st.image(n)
                st.image("D:\\trending\\fashion-dataset\\" + filenames[indices[0][0]])
                # st.header("D:\\trending\\fashion-dataset\\" + filenames[indices[0][0]])
            with col2:
                st.image("D:\\trending\\fashion-dataset\\" + filenames[indices[0][1]])
                # st.header("D:\\trending\\fashion-dataset\\" + filenames[indices[0][1]])
            with col3:
                st.image("D:\\trending\\fashion-dataset\\" + filenames[indices[0][2]])
                # st.header("D:\\trending\\fashion-dataset\\" + filenames[indices[0][2]])
            with col4:
                st.image("D:\\trending\\fashion-dataset\\" + filenames[indices[0][3]])
                # st.header("D:\\trending\\fashion-dataset\\" + filenames[indices[0][3]])
            with col5:
                st.image("D:\\trending\\fashion-dataset\\" + filenames[indices[0][4]])
                # st.header("D:\\trending\\fashion-dataset\\" + filenames[indices[0][4]])
        else:
            st.header("Some error occured in file upload")

if selected == "About":
    st.title(f"Have It!")
    st.markdown("""
        #### Industrial Mentor: Mr. Simran Khara
        #### Faculty Mentor: Mrs. Sonal
        #### By: 
        ######     Ashutosh 19CSU418
        ######     Pankaj 19CSU203
        ######     Pranjal 19CSU217
        ######     Raj 19CSU237
        """)

