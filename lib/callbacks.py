from utils.imports import *



def callback_train():
    """
    Callback to set the global variable when the train button is pressed
    """
    st.session_state.train = 1 
    st.session_state.save = 0 

def callback_save():
    """
    Callback to set the global variable when the save button is pressed
    """
    st.session_state.train= 0
    # It saves the model and the scaler
    with st.spinner("The model is currently saved on your computer..."):  
        print(st.session_state.model_saving_config)
        if "model" in st.session_state.model_saving_config:
            dump(st.session_state.model_saving_config, st.session_state.model_path)
            st.session_state.save = 0
            st.success("The Random forest model has been saved successfully")
        else:
            dump(st.session_state.model_saving_config, st.session_state.scaler_path)
            model = st.session_state.model_scaler_dict["model"]
            print("in the callback")
            print(model)
            print(model.summary())
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
    st.session_state.stop_train = 1

def callback_reset_model():
    st.session_state.model_scaler_dict = dict()
