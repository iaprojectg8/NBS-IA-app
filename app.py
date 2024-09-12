from utils.imports import *
from lib.uploader import *
from lib.visualization import *



# Main execution
uploaded_files = upload_files()
if uploaded_files:
    display_rasters_on_map(uploaded_files)
