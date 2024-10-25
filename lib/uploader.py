from utils.imports import *
from utils.variables import REMAKE_FOLDER, DATAFRAME_HEIGHT
from lib.preprocessing import create_X_y
from lib.nn_models import NNRegressionModel


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
    uploaded_file = st.file_uploader("Chose a model file", type=[".joblib", ".keras", ".h5"], accept_multiple_files=False)
    return uploaded_file

def uplaod_scaler_variable():
    """
    Handle file uploads for model and corresponding scaler
    Returns:
        List of uploaded files.
    """
    st.subheader("Upload a scaler file")
    uploaded_file = st.file_uploader("Chose a scaler file", type=[".joblib"], accept_multiple_files=False)

    return uploaded_file


def load_model_dict(uploaded_file):
    """
    Load the model and the corresponding scaler
    Args:
        uploaded_file (UploadedFile) : Dict containing diverse informations about the file
    Returns:
        model (dict) : The dicitonary loaded contains the model and the corresponing scaler
    """
    model_dict = dict()
    print(uploaded_file)
    if uploaded_file.name.endswith("joblib"):
      
        file = BytesIO(uploaded_file.read())
        model_dict = load(file)

    elif uploaded_file.name.endswith("keras"):
        with tempfile.NamedTemporaryFile(delete=False, suffix='.keras') as temp_file:
            # Write the uploaded file content to the temp file
            temp_file.write(uploaded_file.read())
            temp_file_path = temp_file.name

        model = load_model(temp_file_path)
        print("Removing temporary file...")
        os.remove(temp_file_path)
        uploaded_scaler = uplaod_scaler_variable()
     
        if uploaded_scaler:

            scaler_file = BytesIO(uploaded_scaler.read())
            scaler_dict = load(scaler_file)
            model_dict["model"] = model
            model_dict["scaler"] = scaler_dict["scaler"]
            model_dict["selected_variables"] = scaler_dict["selected_variables"]
        

    return model_dict

def upload_training_file():
    """
    Handle file uploads from the user, to load the corresponding training file.
    Returns:
        CSV training file.
    """
    st.subheader("Upload the current state CSV of the area")
    uploaded_file = st.file_uploader("Choose the current state file", type=["csv"], accept_multiple_files=False)
    return uploaded_file

def upload_test_file():
    """
    Handle file uploads from the user for the CSV test file.
    Returns:
        CSV test file.
    """
    st.subheader("Upload the future state CSV of the area")
    uploaded_file = st.file_uploader("Choose the future state file", type=["csv"], accept_multiple_files=False)
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
    
    file = BytesIO(uploaded_file.read())
    df = pd.read_csv(file)
    
    return df


def check_shape(df):
    """
    Checks if the shape of the provided DataFrame matches the training DataFrame shape stored in session state.

    Args:
        df (pd.DataFrame): DataFrame to check.
    """
    if st.session_state.df_train.shape != df.shape:
        st.write("The current file you chose does not correspond to the future file in terms of shape")
    


def scale_X_current(df_future, df_current, selected_variables, scaler):
    """
    Apply the scaler to the current data 

    Args:
        df_future (pd.DataFrame): Future data DataFrame containing target indices and variables for alignment.
        df_current (pd.DataFrame): Current data DataFrame to be scaled and used for training.
        selected_variables (list): List of variable names to include in X_train.
        scaler (sklearn.preprocessing.StandardScaler): Pre-fitted scaler for scaling the training data.

    Returns:
        np.ndarray: Scaled training data (X_train_scaled).
    """
    # Set the index to the same as the future dataset for both to be comparables
    df_current = df_current.set_index(['LAT', 'LON'])
    df_future = df_future.set_index(['LAT', 'LON'])
    reordered_df = df_current.reindex(df_future.index)
    reordered_df = reordered_df.reset_index()

    {}
    X_current,_ = create_X_y(reordered_df, selected_variables)
    st.dataframe(X_current, height=DATAFRAME_HEIGHT)
    X_current_scaled = scaler.transform(X_current)

    return X_current_scaled