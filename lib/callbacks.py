from utils.imports import *


def callback_train():
    st.session_state.train = 1 
    st.session_state.save = 0 

def callback_save():
    st.session_state.train= 0
    st.session_state.save = 1

def callback_exit():
    st.session_state.train =0
    st.session_state.save = 0