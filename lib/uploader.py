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


#################################### Training page part #############################################################

def upload_csv_train_file():
    """
    Handle file uploads from the user.
    Returns:
        List of uploaded files.
    """
    st.subheader("Upload a CSV train file")
    uploaded_file = st.file_uploader("Choose CSV file", type=["csv"], accept_multiple_files=False)
    return uploaded_file



################################### Testing page part ################################################################

def upload_model_file():
    """
    Handle file uploads for model and corresponding scaler
    Returns:
        List of uploaded files.
    """
    st.subheader("Upload a model file")
    uploaded_file = st.file_uploader("Chose a model file", type=[".joblib"], accept_multiple_files=False)
    return uploaded_file


def load_model(uploaded_file):
    """
    Load the model and the corresponding scaler
    Args:
        uploaded_file (UploadedFile) : Dict containing diverse informations about the file
    Returns:
        model (dict) : The dicitonary loaded contains the model and the corresponing scaler
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=".joblib") as temp_file:
        temp_file.write(uploaded_file.read())
        temp_file_path = temp_file.name
    model_scaler = load(temp_file_path)

    return model_scaler

def upload_training_file():
    """
    Handle file uploads from the user, to load the corresponding training file.
    Returns:
        CSV training file.
    """
    st.subheader("Upload your training file")
    uploaded_file = st.file_uploader("Choose your training CSV file", type=["csv"], accept_multiple_files=False)
    return uploaded_file

def upload_test_file():
    """
    Handle file uploads from the user for the CSV test file.
    Returns:
        CSV test file.
    """
    st.subheader("Upload a test CSV file")
    uploaded_file = st.file_uploader("Choose CSV file", type=["csv"], accept_multiple_files=False)
    return uploaded_file

################################### This is used for all the parts ########################################################

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