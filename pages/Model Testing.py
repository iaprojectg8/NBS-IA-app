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
    st.write(f"Model used: `{st.session_state.model_scaler_dict["model"]}`")
    st.write(f"Scaler used: `{str(st.session_state.model_scaler_dict["scaler"])}`")

else:
    uploaded_model = upload_model_file()
    if uploaded_model:
        st.session_state.model_scaler_dict = load_model(uploaded_file=uploaded_model)


# Simplify the variable
if "model" in st.session_state.model_scaler_dict and "scaler" in st.session_state.model_scaler_dict:
    model = st.session_state.model_scaler_dict["model"]
    scaler = st.session_state.model_scaler_dict["scaler"]


uploaded_test_file = upload_test_file()
if uploaded_test_file:
        
    df = manage_csv(uploaded_file=uploaded_test_file) 
    selected_variables = st.multiselect("Chose the variable on which you want to train", options=TRAINING_LIST,default=model.feature_names_in_)
    X,y = create_X_y(df,selected_variables) 
    st.subheader("Testing dataframe")
    st.dataframe(X, height=DATAFRAME_HEIGHT)

    # Scale the input data with the same scaler as during the training process 
    X_scaled = scaler.transform(X)

    if st.session_state.df_train is not None:
        st.write("Your training dataframe is already in memory so you don't need to upload it")
        
    else:
        uploaded_training_file = upload_training_file()
        if uploaded_training_file:
            st.session_state.df_train = manage_csv(uploaded_file=uploaded_training_file)
        
    
    if st.session_state.df_train is not None:
        st.subheader("Training dataset")

        # Manage the case where the indices are not the same in train and test dataset in order to compare them
        df_current = st.session_state.df_train.set_index(['LAT', 'LON'])
        df_future = df.set_index(['LAT', 'LON'])
        reordered_df = df_current.reindex(df_future.index)
        reordered_df = reordered_df.reset_index()
    
        X_train,y_train = create_X_y(reordered_df, selected_variables)
        st.dataframe(X_train, height=DATAFRAME_HEIGHT)
        X_train_scaled = scaler.transform(X_train)
       
    st.button("Test your model", on_click=callback_test)

    if st.session_state.test:
        map = leafmap.Map()
        y_pred, y_train_pred = test(X_scaled, X_train_scaled, model=model)

        # Creating new_fields
        df["LST_pred"] = y_train_pred
        df["LST_pred_fut"] = y_pred
        df["Diff_LST"] =  y_pred - y_train_pred

        # Show the map with all the raster when they are created
        with st.status("Creating the rasters...",expanded=True):
            st.write("Current LST prediction...")
            create_raster(df=df, variable="LST_pred",map=map)
            st.write("Future LST prediction...")
            create_raster(df=df, variable="LST_pred_fut",map=map)
            st.write("Difference between both...")
            create_raster(df=df, variable="Diff_LST", map=map)
            map.to_streamlit()
     
