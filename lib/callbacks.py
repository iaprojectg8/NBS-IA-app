from utils.imports import *



def callback_train():
    """
    Callback to set the global variable when the train button is pressed
    """
    st.session_state.train = 1 
    st.session_state.save = 0 

def callback_save():
    """
    Callback to set the global variable when the save button is pressed. Two casev are handled here.
    Either the model is a Random Forest and then we can save the model and all the others features with
    joblib this tool allows it.
    Either the model is a neural network and then we need to use the keras model saver and joblib as well 
    for the scaler part.
    """
    st.session_state.train= 0
    # It saves the model and the scaler
    with st.spinner("The model is currently saved on your computer..."):  
        if "model" in st.session_state.model_saving_config:
            dump(st.session_state.model_saving_config, st.session_state.model_path)
            st.session_state.save = 0
            st.success("The Random forest model has been saved successfully")
        else:
            dump(st.session_state.model_saving_config, st.session_state.scaler_path)
            model = st.session_state.model_scaler_dict["model"]
            model.save(st.session_state.model_path)
            st.success("The Neural Network model has been saved successfully and the sclaer as well")

def callback_exit():
    """
    Callback to set the global variable when the exit button is pressed
    """
    st.session_state.train = 0
    st.session_state.save = 0

def callback_test():
    """
    Callback to set the global variable when the test button is pressed
    """
    st.session_state.test = 1

def callback_stop_train():
    """
    Callback that allows the user to stop the train
    """
    st.session_state.stop_train = 1

def callback_reset_model():
    """
    Callback to reset the model used in the test part
    """
    st.session_state.model_scaler_dict = dict()
