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
    st.session_state.save = 1

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