from utils.imports import *
from lib.uploader import *
from lib.visualization import *
from streamlit import config_option


# Main execution
uploaded_files = upload_files()
if uploaded_files:
    print(uploaded_files)
    manage_uploaded_files(uploaded_files)
