from utils.imports import *
from lib.session_variables import *
from lib.uploader import *
from lib.visualization import *
from lib.model import rdf_regressor, test, create_raster
from streamlit import config_option
from lib.callbacks import *
from utils.variables import G8_LOGO_PATH, KERAN_LOGO_PATH, DATAFRAME_HEIGHT
from lib.preprocessing import *
from lib.tools import put_logo_if_possible
from lib.logo_style import increase_logo
        
put_logo_if_possible()
st.logo(G8_LOGO_PATH)
increase_logo()

st.title("Model Tester")


#### Model part

if st.session_state.model_scaler_dict is not None:
    st.write("You don't have to load your model, it is already in memory and here is a brief description of it")
    st.write(str(st.session_state.model_scaler_dict["model"]))
else:
    uploaded_model = upload_model_file()
    if uploaded_model:
        st.session_state.model_scaler_dict = load_model(uploaded_file=uploaded_model)



model = st.session_state.model_scaler_dict["model"]
scaler = st.session_state.model_scaler_dict["scaler"]
#### CSV part   
uploaded_test_file = upload_test_file()
if uploaded_test_file:
        
    print(st.session_state.selected_variables)
    df = manage_csv(uploaded_file=uploaded_test_file) 
    selected_variables = selected_variables = st.multiselect("Chose the variable on which you want to train", options=TRAINING_LIST,default=st.session_state.selected_variables)
    X,y = create_X_y(df,selected_variables) 
    st.subheader("Testing dataframe")
    st.dataframe(X, height=DATAFRAME_HEIGHT)
    X_scaled = scaler.transform(X)

    if st.session_state.df_init is not None:
        st.write("Your training dataframe is already in memory so you don't need to upload it")
        
    else:
        st.session_state.df_init = upload_training_file()
    
    if st.session_state.df_init is not None:
        st.subheader("Training dataset")
        df_current = st.session_state.df_init.set_index(['LAT', 'LON'])
        df_future = df.set_index(['LAT', 'LON'])

        # Reindex df2 to match the order of df1
        reordered_df = df_current.reindex(df_future.index)
        print(df)
        reordered_df = reordered_df.reset_index()
        print(reordered_df)

        # Optionally, reset the index if you don't want 'lat' and 'lon' as index
        
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

        print(df)
        with st.status("Creating the rasters...",expanded=True):
            st.write("Current LST prediction...")
            map = create_raster(df=df, variable="LST_pred",map=map)
            st.write("Future LST prediction...")
            map = create_raster(df=df, variable="LST_pred_fut",map=map)
            st.write("Difference between both...")
            map = create_raster(df=df, variable="Diff_LST", map=map)
            map.to_streamlit()
     
