from utils.imports import *
from utils.variables import KERAN_LOGO_PATH

def put_logo_if_possible():
    with open(KERAN_LOGO_PATH, "rb") as file:
        svg_content = file.read()
    st.set_page_config(page_title="IA Tool",page_icon=svg_content)

def increase_logo():

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