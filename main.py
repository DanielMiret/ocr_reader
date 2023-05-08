from PIL import Image
from ocr_reader import OCRReader

import streamlit as st
import torch
import numpy as np


if __name__ == "__main__":
    st.title('Python Image Reader')

    # tensor = torch.rand(50, 3)
    # st.code(tensor)

    st.subheader(
        body=":mechanical_arm: Hardware Acceleration",
        anchor=False,
        help="Select an engine to run. CPU is slower than CUDA"
    )
    cuda_status = torch.cuda.is_available()
    options = ['CPU', 'CUDA'] if cuda_status else ['CPU']
    tensor_option = st.radio(
        label='Tensor Engine:',
        options=options,
        horizontal=True
    )
    gpu: bool = True if tensor_option == "CUDA" else False
    if tensor_option == "CUDA":
        st.image(
            image="utils/gpu.png",
            caption=torch.cuda.get_device_name(0),
            width=105
        )

    st.subheader(
        body=":open_file_folder: Upload an image",
        anchor=False,
        help="Select an image file to upload"
    )
    file = st.file_uploader(
        label="Upload an image file to analyse",
        type=["png"],
        label_visibility="collapsed"
    )

    if file is not None:
        st.header(
            body=":mag_right: Result",
            anchor=False
        )
        col1, col2 = st.columns(2)
        img = Image.open(file)
        with col1:
            st.image(img, caption='Original')

        text: str = ""
        with col2:
            with st.spinner(text='Reading...'):
                text: str | None = OCRReader(
                    img=np.array(img),
                    language=["pt", "en"],
                    gpu=gpu,
                    new_img_path="temp/new.png"
                ).easy_ocr()

            st.image("temp/new.png", caption='Text Detection',
                     output_format="PNG")

        if text:
            st.header(
                body=":mag_right: Text Result",
                anchor=False
            )
            st.code(text)
