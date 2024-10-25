from utils.imports import *
from lib.session_variables import *
from lib.uploader import *
from lib.model_test import test, create_raster, init_test
from streamlit import config_option
from lib.callbacks import *
from utils.variables import G8_LOGO_PATH, DATAFRAME_HEIGHT
from lib.preprocessing import *
from lib.logo_style import increase_logo, put_logo_if_possible
        
put_logo_if_possible()
st.logo(G8_LOGO_PATH)
increase_logo()

st.title("Model Tester")
init_test()
# Information given about the model trained or uploaded
if st.session_state.model_scaler_dict:
    st.write("You don't have to load your model, it is already in memory and here is a brief description of it")
    if "model" in st.session_state.model_scaler_dict:
        st.write(f"Model used: `{st.session_state.model_scaler_dict["model"]}`")
    
    st.write(f"Scaler used: `{str(st.session_state.model_scaler_dict["scaler"])}`")
    st.button("Reset a model", on_click=callback_reset_model)

else:
    uploaded_model = upload_model_file()
    if uploaded_model:
        
        st.session_state.model_scaler_dict = load_model_dict(uploaded_file=uploaded_model)

# Simplify the variable
uploaded_test_file = None
if "model" in st.session_state.model_scaler_dict and "scaler" in st.session_state.model_scaler_dict:
    model = st.session_state.model_scaler_dict["model"]
    scaler = st.session_state.model_scaler_dict["scaler"]
    train_variables = st.session_state.model_scaler_dict["selected_variables"]

    uploaded_test_file = upload_test_file()
if uploaded_test_file:
        
    df_future = manage_csv(uploaded_file=uploaded_test_file) 
    selected_variables = st.multiselect("Chose the variable on which you want to train", options=TRAINING_LIST,default=train_variables)
    X,y = create_X_y(df_future,selected_variables) 
    st.subheader("Future dataset")
    st.dataframe(X, height=DATAFRAME_HEIGHT)

    # Scale the input data with the same scaler as during the training process 
    X_scaled = scaler.transform(X)

    df_current= pd.DataFrame()
    uploaded_training_file = upload_training_file()

    if uploaded_training_file:

        df_current = manage_csv(uploaded_file=uploaded_training_file)
        st.subheader("Current dataset")
    
        if df_current.shape == df_future.shape:

            X_train_scaled = scale_X_current(df_future, df_current, selected_variables, scaler)
        else :
            st.warning("The current file you chose does not correspond to the future file in terms of shape  \n\
                    To be able to test your model you need to have both current and future CSV file of the zone")
            
        
        st.button("Test your model", on_click=callback_test)
    

        if st.session_state.test and df_future.shape == df_current.shape:
            map = leafmap.Map()


            # Show the map with all the raster when they are created
            with st.status("Creating the rasters...",expanded=True):

                # Making the 
                st.write("Making the prediction on current and the future set to compare them")
                y_pred, y_train_pred = test(X_scaled, X_train_scaled, model=model)
                # Creating new_fields
                st.write("Inserting prediction in the main dataframe")
                df_future["LST_pred"] = y_train_pred
                df_future["LST_pred_fut"] = y_pred
                df_future["Diff_LST"] =  y_pred - y_train_pred
              
                st.write("Current LST prediction...")
                create_raster(df=df_future, variable="LST_pred",map=map)
                st.write("Future LST prediction...")
                create_raster(df=df_future, variable="LST_pred_fut",map=map)
                st.write("Difference between both...")
                create_raster(df=df_future, variable="Diff_LST", map=map)
                map.to_streamlit()
        
