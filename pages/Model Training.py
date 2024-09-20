from utils.imports import *
from lib.session_variables import *
from lib.uploader import *
from lib.visualization import *
from lib.model import rdf_regressor, save_model, display_all_precedent_training_graphs
from streamlit import config_option
from lib.callbacks import *
from utils.variables import G8_LOGO_PATH, TRAINING_LIST, DATAFRAME_HEIGHT
from lib.preprocessing import *
from lib.tools import put_logo_if_possible
from lib.logo_style import increase_logo
from copy import copy, deepcopy
    
put_logo_if_possible()
st.logo(G8_LOGO_PATH)
increase_logo()

st.title("Model trainer")
st.session_state.csv_file = upload_csv_train_file()
if st.session_state.csv_file:
    

    df = manage_csv(uploaded_file=st.session_state.csv_file)
    print(df)
    st.session_state.df_init = df
    n_rows = len(df)
    print(st.session_state.selected_variables)
    selected_variables = st.multiselect("Chose the variable on which you want to train", options=TRAINING_LIST,default=st.session_state.selected_variables)
    st.session_state.selected_variables = selected_variables
    print(st.session_state.selected_variables)
    # This selectbox is made to chose the amount of data to train, divided by 10 and 2 the orginial amount of data
    data_amount = st.selectbox("Choose the data size for training",[int(n_rows/10**(i/2)) if i%2==0 else int(n_rows/(10**((i-1)/2)*2),) for i in range(10)])
    df_sampled = copy(df).sample(data_amount,ignore_index=True)
    
    # Separate the labels and the variables
    X,y = create_X_y(df_sampled,copy(selected_variables))
    st.subheader("Training dataframe")
    st.dataframe(X, height=DATAFRAME_HEIGHT)
    print("AFter creating X and y",st.session_state.selected_variables)


    estimator = st.number_input(label="Amount of estimators (trees)", min_value=1, max_value=150, value=25)
    test_size = st.slider(label="Chose the data proportion given to test",min_value=0.0, max_value=1.0, step=0.01, value=0.2)
    st.button("Train a model", on_click=callback_train)

    if st.session_state.train:
        
        model_scaler_dict = rdf_regressor(X,y, estimator, test_size)

        print(st.session_state.input_path)
        st.write("Your model is in memory you can go to the test page and try to predict on a new dataset containing the same variable as the ")
        save_model()
        st.session_state.training_done = 1
        
    elif st.session_state.training_done:
        display_all_precedent_training_graphs()
        
    if st.session_state.save:
         
      
        dump(model_scaler_dict, st.session_state.input_path)
        st.session_state.save = 0



    