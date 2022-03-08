import streamlit as st
from streamlit_cropper import st_cropper
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib import style
style.use('fivethirtyeight')
st.set_option('deprecation.showfileUploaderEncoding', False)

from model import model, preprocess, CLASSES

# Upload an image and set some options for demo purposes
st.header("Traffic sign recognition")
img_file = st.sidebar.file_uploader(label='Upload a file', type=['png', 'jpg'])
realtime_update = st.sidebar.checkbox(label="Update in Real Time", value=True)
box_color = st.sidebar.color_picker(label="Box Color", value='#0000FF')
aspect_choice = st.sidebar.radio(label="Aspect Ratio", options=["1:1", "16:9", "4:3", "2:3", "Free"])
aspect_dict = {
    "1:1": (1, 1),
    "16:9": (16, 9),
    "4:3": (4, 3),
    "2:3": (2, 3),
    "Free": None
}
aspect_ratio = aspect_dict[aspect_choice]

c1, c2 = st.columns(2)

if img_file:
    img = Image.open(img_file)
    w, h = img.size
    if w<100: img = img.resize((w*3, h*3))

    if not realtime_update:
        st.write("Double click to save crop")
    # Get a cropped image from the frontend
    cropped_img = st_cropper(img, realtime_update=realtime_update, box_color=box_color,
                                aspect_ratio=aspect_ratio)
    
    # Manipulate cropped image at will
    _ = cropped_img.thumbnail((200,200))
    c2.image(cropped_img)
    # c2.write("Input image")

    topk = 10    

    bx = preprocess(cropped_img)
    ps = model.predict(bx)[0]
    label = CLASSES[model.predict_classes(bx)[0]]

    names = np.array([CLASSES[i] for i in range(len(CLASSES))])
    topk_idxs = np.argsort(ps)[:topk]
    topk_preds = names[topk_idxs][:topk]
    topk_scores = ps[topk_idxs][:topk]

    fig = plt.figure(figsize=(10,5))
    plt.barh(topk_preds, topk_scores)
    plt.title(f"Best Prediction: {label}")
    plt.yticks(fontsize=24)
    plt.xticks(fontsize=24, rotation=90)
    c1.pyplot(fig)