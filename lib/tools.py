from utils.imports import *
from utils.variables import KERAN_LOGO_PATH

def put_logo_if_possible():
    with open(KERAN_LOGO_PATH, "rb") as file:
        svg_content = file.read()
    st.set_page_config(page_title="IA Tool",page_icon=svg_content)