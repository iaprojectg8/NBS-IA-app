from utils.imports import *

def upload_files():
    """
    Handle file uploads from the user.
    Returns:
        List of uploaded files.
    """
    st.title("Upload Raster Files")
    uploaded_files = st.file_uploader("Choose TIFF files", type=["tif", "tiff"], accept_multiple_files=True)
    return uploaded_files