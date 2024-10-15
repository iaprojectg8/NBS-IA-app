from utils.imports import *
from utils.variables import  TRAINING_LIST


# Initialize the input path to save the model

if "model_path" not in st.session_state:
    
    if "model" not in os.listdir(os.getcwd()):
        os.mkdir("model") 
    st.session_state.model_path = str()

if "scaler_path" not in st.session_state:
    st.session_state.scaler_path = str()

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

if "loss_history" not in st.session_state:
    st.session_state.loss_history = None

if "mae_history" not in st.session_state:
    st.session_state.mae_history = None

if "lr_history" not in st.session_state:
    st.session_state.lr_history = None

st.session_state.lr_history
# Session varialbe useful to know if a train process has already been done to display the graphs

if "training_done" not in st.session_state: 
    st.session_state.training_done = 0

# Session variable useful to keep the train dataset in memory for the test part

if "df_train" not in st.session_state:
    st.session_state.df_train = None


# New variable that came with the arrival of the neural net implementation
if "stop_train" not in st.session_state:
    st.session_state.stop_train = 0

if "history" not in st.session_state:
    st.session_state.history = []

if "model" not in st.session_state:
    st.session_state.model = None

if "model_saving_config" not in st.session_state:
    st.session_state.model_saving_config = dict()
