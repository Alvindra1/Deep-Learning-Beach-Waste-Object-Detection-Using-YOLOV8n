import cv2 
import streamlit as st
import numpy as np
from ultralytics import YOLO
from PIL import Image

# model and others

@st.cache_resource
def load_model():
    model = YOLO("./model/best.pt")
    return model

def detection(image,model):
    results = model(image)

    annotation = results[0].plot()
    annotation = cv2.cvtColor(annotation,cv2.COLOR_BGR2RGB)
    
    return annotation

# page section
st.set_page_config(
    page_title="Beach Waste Object Detection",
    layout="wide"
)


page = st.sidebar.radio("Page",("Main","About"))

if page == "Main":
    st.markdown("<h1 style='text-align:center'>Beach Waste Detection Using YOLOV8</h1>",unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    
    st.info("Upload an image")

    upload_image = st.file_uploader("Upload an image", type=['jpg','jpeg','png'])


    col1,col2 = st.columns(2)

    before_img = None
    after_img = None

    if upload_image is not None:
        before_img = Image.open(upload_image)
        with col1:
            st.subheader("Before Image")
            st.image(before_img)

        with col2:
            st.subheader("After Detection")

            with st.spinner("Detecting objects"):
                try:
                    model = load_model()
                    detect_img = detection(before_img, model)
                    st.image(detect_img,use_container_width=True)
                except Exception as e:
                    st.error(f"Error: {str(e)}")
    else:
        st.info(f"Please upload an image to be detected")

    st.subheader("How to use the model")
    st.markdown("First upload the image that you want to detect by clicking the Browse files button")
    st.markdown("pick an image that want to be detected")
    st.markdown("then wait for the model to detect the image that was uploaded")
    # st.markdown("then you can expect the output like the example below")

    


elif page =="About":
    st.markdown("<h1 style='text-align:center'>About page</h1>",unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    # project details(model,library use etc)
    st.markdown("<h2>Project Details</h2)", unsafe_allow_html=True)
    st.markdown("This project is to make an AI to be able to detect waste in beaches. Utilizing YOLOV8n a deep learning model. The model will analyze the images to detect any kind of waste/trash scattered across the beach.")
    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("<h2>Github</h2)", unsafe_allow_html=True)
    st.markdown("Github link: https://github.com/Alvindra1/Deep-Learning-Beach-Waste-Object-Detection-Using-YOLOV8n")
    st.markdown("<br>", unsafe_allow_html=True)

    # st.markdown("<h2>Project description</h2)", unsafe_allow_html=True)
    # st.markdown("")
    # st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("<h2>Technologies Used</h2>",unsafe_allow_html=True)
    st.markdown("YOLOV8n - YOLOV8 nano version")
    st.markdown("Streamlit - Web app")
    st.markdown("OpenCV - Computer Vision library")
    st.markdown("Ultralytics - YOLO library")
    st.markdown("Pytorch - A deep learning library")
    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("<h2>Dataset</h2>",unsafe_allow_html=True)
    st.markdown("The dataset that was use was BePLi Dataset v1: Beach Plastic Litter Dataset Version 1")
    # st.markdown("The dataset consist of 4 folder(original images, train folder, test folder, and validation folder) 3708 total images, 2226 train images, 741 test images, and 742 validation images. ")
    st.markdown("The dataset consist of 4 folders:")
    st.markdown("- Original forlder contains 3708 images")
    st.markdown("- Train forlder contains 2226 images")
    st.markdown("- Testing forlder contains 741 images")
    st.markdown("- Validation forlder contains 742 images")
    st.markdown("website link: https://www.seanoe.org/data/00811/92297/")
    st.markdown("<br>", unsafe_allow_html=True)


    
    st.markdown("<h2 style='text-align:center'>Group Members</h2>",unsafe_allow_html=True)

    g1,g2,g3 = st.columns(3)


    with g1:
        st.markdown("""
                    <div style='padding:20px; border-radius: 10px; text-align: center;'>
                    <h3 style='color:white;'>Alvindra Angga Sukalaksana</h>
                    """, unsafe_allow_html=True)
    with g2:
        st.markdown("""
                    <div style='padding:20px; border-radius: 10px; text-align: center;'>
                    <h3 style='color:white;'>Andersen Chandra</h>
                    """, unsafe_allow_html=True)
    with g3:
        st.markdown("""
                    <div style='padding:20px; border-radius: 10px; text-align: center;'>
                    <h3 style='color:white;'>Victor Ra Thenisius</h>
                    """, unsafe_allow_html=True)



    















