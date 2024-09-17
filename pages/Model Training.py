from utils.imports import *
from lib.session_variables import *
from lib.uploader import *
from lib.visualization import *
from lib.model import rdf_regressor
from streamlit import config_option
from lib.callbacks import *
from utils.variables import G8_LOGO_PATH, KERAN_LOGO_PATH
from lib.preprocessing import *
from lib.tools import put_logo_if_possible
    
put_logo_if_possible()

st.logo(G8_LOGO_PATH)
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

st.title("Model trainer")
# Main execution
uploaded_files = upload_train_file_model()
if uploaded_files:
    print(uploaded_files)
    df, selected_variables = manage_uploaded_model(uploaded_files)
    df = df.sample(10000)
    X,y = create_X_y(df,selected_variables)
    estimator = st.number_input(label="Amount of estimators (trees)", min_value=1, max_value=150, value=25)
    st.button("Train a model", on_click=callback_train)
    if st.session_state.train:
        
        rdf_regressor(X, y, estimator=estimator)

    