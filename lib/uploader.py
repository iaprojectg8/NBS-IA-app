from utils.imports import *
from utils.variables import REMAKE_FOLDER, DATAFRAME_HEIGHT


################################ Part exclusively for the raster viz page ###########################################

def upload_files_raster_viz():
    """
    Handle file uploads from the user.
    Returns:
        List of uploaded files.
    """
    st.subheader("Upload Raster or CSV Files")
    uploaded_files = st.file_uploader("Choose TIFF or CSV files", type=["tif", "tiff", "csv"], accept_multiple_files=True)
    return uploaded_files




#################################### Training page part ##############################################

def upload_csv_train_file():
    """
    Handle file uploads from the user.
    Returns:
        List of uploaded files.
    """
    st.subheader("Upload a CSV train file")
    uploaded_file = st.file_uploader("Choose CSV file", type=["csv"], accept_multiple_files=False)
    return uploaded_file


def upload_training_file():
    """
    Handle file uploads from the user.
    Returns:
        List of uploaded files.
    """
    st.subheader("Upload your training file")
    uploaded_file = st.file_uploader("Choose your training CSV file", type=["csv"], accept_multiple_files=False)
    return uploaded_file

def upload_test_file():
    """
    Handle file uploads from the user.
    Returns:
        List of uploaded files.
    """
    st.subheader("Upload a test CSV file")
    uploaded_file = st.file_uploader("Choose CSV file", type=["csv"], accept_multiple_files=False)
    return uploaded_file

def upload_model_file():
    """
    Handle file uploads from the user.
    Returns:
        List of uploaded files.
    """
    st.subheader("Upload a model file")
    uploaded_file = st.file_uploader("Chose a model file", type=[".joblib"], accept_multiple_files=False)
    return uploaded_file



#### Faire une liste globale qui puisse être mise en variable globale.

def manage_test_variable_selection(csv_file):
    if csv_file.type=="application/vnd.ms-excel":
        df = manage_csv(uploaded_file=csv_file)
        variable_list = ["LAT", "LON","LS1","LS2","LS3","LS4","LS5","LS6","OCCSOL","URB","ALT","EXP","PENTE","NATSOL","NATSOL2","HAUTA","CATHYD","ZONECL","ALB"]
        selected_variables = st.multiselect("Chose the variable on which you want to train", options=variable_list,default=st.session_state.selected_variables)

    return df, selected_variables

            


def load_model(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".joblib") as temp_file:
        temp_file.write(uploaded_file.read())
        temp_file_path = temp_file.name
    model = load(temp_file_path)

    return model


################################### This is used for all the parts ##################################
def manage_csv(uploaded_file):
    """
    Manage the opening of the CSV file from the temporary file
    Args:
        uploaded_file (UploadedFile) : Dict containing diverse informations about the file
    Returns:
        df (pd.Dataframe) : dataframe containing all the variables
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as temp_file:
        temp_file.write(uploaded_file.read())
        temp_file_path = temp_file.name
    df = pd.read_csv(temp_file_path)
    
    return df