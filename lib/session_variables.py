from utils.imports import *
from utils.variables import MODEL_FILE

if "train" not in st.session_state:
    st.session_state.train = 0

if "model" not in st.session_state:
    st.session_state.model = None


if "input_path" not in st.session_state:
    working_dir = os.getcwd()
    model_path = os.path.join(working_dir, "model")
    if "model" not in os.listdir(working_dir):
        os.mkdir("model")        
    
    st.session_state.input_path = os.path.join(model_path, MODEL_FILE)

if "model_path" not in st.session_state:
    st.session_state.model_path = st.session_state.input_path

if "save" not in st.session_state:
    st.session_state.save=0