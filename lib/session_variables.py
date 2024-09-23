from utils.imports import *
from utils.variables import MODEL_FILE, TRAINING_LIST

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


if "selected_variables" not in st.session_state:
    st.session_state.selected_variables=TRAINING_LIST

if "estimator" not in st.session_state:
    st.session_state.estimator = 50

if "test"  not in st.session_state:
    st.session_state.test = 0


if "scaler" not in st.session_state:
    st.session_state.scaler = None
    
if "model_scaler_dict" not in st.session_state:
    st.session_state.model_scaler_dict = dict()
    
if "stat_on_pred_fig1" not in st.session_state:    
    st.session_state.stat_on_pred_fig1 = None


if "stat_on_pred_fig2" not in st.session_state:    
    st.session_state.stat_on_pred_fig2 = None
    
if "results_fig1" not in st.session_state:    
    st.session_state.results_fig1 = None
    
if "results_fig2" not in st.session_state:    
    st.session_state.results_fig2 = None
    
if "training_done" not in st.session_state: 
    st.session_state.training_done = 0
    
if "csv_file" not in st.session_state:
    st.session_state.csv_file = 0

if "df_train" not in st.session_state:
    st.session_state.df_train = None