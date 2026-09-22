import streamlit as st
from PIL import Image


st.set_page_config(
    page_title="ChessVision",
    page_icon="♟",
    layout="wide",
)


st.title("♟ ChessVision")
st.write(
    "Convert a physical chessboard image into a digital chess position."
)

st.divider()


# -----------------------------
# Input & Controls
# -----------------------------

st.subheader("Upload Chessboard")

uploaded_file = st.file_uploader(
    "Upload a chessboard image",
    type=["png", "jpg", "jpeg"],
)


if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.image(
        image,
        caption="Uploaded chessboard",
        use_container_width=True,
    )

    st.success("Image uploaded successfully.")

else:

    st.info(
        "Upload a PNG, JPG, or JPEG image of a chessboard to begin."
    )