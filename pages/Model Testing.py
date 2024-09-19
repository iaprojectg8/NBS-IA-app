from utils.imports import *
from lib.session_variables import *
from lib.uploader import *
from lib.visualization import *
from lib.model import rdf_regressor, test, create_raster
from streamlit import config_option
from lib.callbacks import *
from utils.variables import G8_LOGO_PATH, KERAN_LOGO_PATH
from lib.preprocessing import *
from lib.tools import put_logo_if_possible
from lib.logo_style import increase_logo
        
put_logo_if_possible()

st.logo(G8_LOGO_PATH)
increase_logo()

st.title("Model Tester")


#### Model part

if st.session_state.model is not None:
    st.write("You don't have to load your model, it is already in memory and here is a brief description of it")
    st.write(str(st.session_state.model))
else:
    uploaded_model = upload_model_file()
    if uploaded_model:
        st.session_state.model = load_model(uploaded_file=uploaded_model)


#### CSV part   
uploaded_files = upload_test_file_model()
if uploaded_files:
        
    print(uploaded_files)
    print(st.session_state.selected_variables)
    df, selected_variables = manage_uploaded_model(uploaded_files)
    X,y = create_X_y(df,selected_variables) 
    X_scaled = st.session_state.scaler.transform(X)
    st.button("Test your model", on_click=callback_test)
    if st.session_state.test:
        map = leafmap.Map()
        y_pred = test(X_scaled, st.session_state.model)

        # Creating new_fields
        df["LST_pred"] = y_pred
        df["Diff_LST"] =  y_pred - y

        print(df)
        map = create_raster(df=df, variable="LST",map=map)
        map = create_raster(df=df, variable="LST_pred",map=map)
        map = create_raster(df=df, variable="Diff_LST", map=map)
        map.to_streamlit()
        # show(difference)
