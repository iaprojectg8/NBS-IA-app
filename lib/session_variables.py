from utils.imports import *
from utils.variables import MODEL_FILE, TRAINING_LIST


# Initialize the input path to save the model

if "input_path" not in st.session_state:
    working_dir = os.getcwd()
    model_path = os.path.join(working_dir, "model")
    if "model" not in os.listdir(working_dir):
        os.mkdir("model")        
    
    st.session_state.input_path = os.path.join(model_path, MODEL_FILE)

# Session variables to keep the model in memory

if "selected_variables" not in st.session_state:
    st.session_state.selected_variables=TRAINING_LIST

if "estimator" not in st.session_state:
    st.session_state.estimator = 50
    
if "model_scaler_dict" not in st.session_state:
    st.session_state.model_scaler_dict = dict()


# Global variables to know if the corresponding button has been clicked

if "train" not in st.session_state:
    st.session_state.train = 0

if "save" not in st.session_state:
    st.session_state.save=0


if "test"  not in st.session_state:
    st.session_state.test = 0


# Global variable to display training graphs

if "stat_on_pred_fig1" not in st.session_state:    
    st.session_state.stat_on_pred_fig1 = None

if "stat_on_pred_fig2" not in st.session_state:    
    st.session_state.stat_on_pred_fig2 = None
    
if "results_fig1" not in st.session_state:    
    st.session_state.results_fig1 = None
    
if "results_fig2" not in st.session_state:    
    st.session_state.results_fig2 = None


# Session varialbe useful to know if a train process has already been done to display the graphs

if "training_done" not in st.session_state: 
    st.session_state.training_done = 0

# Session variable useful to keep the train dataset in memory for the test part

if "df_train" not in st.session_state:
    st.session_state.df_train = None