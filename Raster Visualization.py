from utils.imports import *
from lib.uploader import *
from lib.visualization import *
from streamlit import config_option
from lib.callbacks import *


st.logo("logo/Logo_G8.png")
st.markdown("""
            <style>
    div[data-testid="stSidebarHeader"] > img, div[data-testid="collapsedControl"] > img {
      height: 3rem;
      width: auto;
    }
  
    div[data-testid="stSidebarHeader"], div[data-testid="stSidebarHeader"] > *,
    div[data-testid="collapsedControl"], div[data-testid="collapsedControl"] > * {
      display: flex;
      align-items: center;
    }
</style>
            """,unsafe_allow_html=True)

# Main execution
st.title("Raster Visualizer")
uploaded_files = upload_files_raster_viz()
if uploaded_files:
    print(uploaded_files)
    manage_uploaded_files_raster_viz(uploaded_files)
    # st.button("Train a predictive model", on_click=whatever_callback)
