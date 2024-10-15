from utils.imports import *
from utils.variables import *

def clean_csv(filename):
    """
    Function to clean CSV when the QGIS algo was not working as
    expected
    """
    df = pd.read_csv(filename,encoding='ISO-8859-1')
    print(df)
    df = df.drop(labels=["layer", "path"], axis=1)
    print(df)
    df.to_csv("Yaounde_Futur_Urb.csv",index=False)


def create_X_y(df,parameters_list):
    """Prepares feature matrix X and target vector y from the DataFrame by selecting the right parameters
    and cleaning the data.
    
    Args:
        df (DataFrame): The input DataFrame.
        parameters_list (list): List of parameters to include in the feature matrix.
    
    Returns:
        tuple: Feature matrix X and target vector y.
    """
    # y will always be the same because this is the variable we want to predict
    
    df = take_right_parameters(df,parameters_list)
    y = df["LST"] 
    y = np.array(y,dtype=np.float16)     
    X = df.drop('LST', axis=1)  # If you want to take all the variables except the dependant one do this, even if it is hard to understand
    
    return X,y


def take_right_parameters(df,params_to_take=[]):
    """
    Selects or drops specified parameters (columns) from the DataFrame.
    
    Args:
        df (DataFrame): The input DataFrame.
        params_to_take (list): List of parameters to include in the DataFrame.

    Returns:
        DataFrame: The DataFrame with the specified parameters taken or dropped.
    """
    if params_to_take!=[]:
        if "LST" not in params_to_take:
            params_to_take.append("LST")
        df = df[params_to_take]
            
    return df

def class_label_dicts(df):
    """
    Creates dictionaries to map between class labels and their corresponding numerical representations.

    Args:
    df (DataFrame): Input DataFrame.

    Returns:
    dict: Dictionary mapping labels to classes.
    dict: Dictionary mapping classes to labels.
    """
    labels = df["LST"]
    unique_labels = pd.unique(labels)
    unique_labels_sorted = np.sort(unique_labels)
    

    classes_to_labels = dict()
    labels_to_classes = dict()

    for i, label in enumerate(unique_labels_sorted):
        label = round(label,2)
        classes_to_labels[i] = label
        labels_to_classes[label] = i

   
    # classes = [labels_to_classes[round(label,2)] for label in labels]
    # df["LST"] = classes

    return labels_to_classes, classes_to_labels



def label_to_class(df,labels_to_classes):
    """
    Converts class labels in the "LST" column of the DataFrame to their corresponding numerical representations.

    Args:
    df (DataFrame): Input DataFrame.
    labels_to_classes (dict): Dictionary mapping labels to classes.

    Returns:
    DataFrame: DataFrame with class labels converted to numerical representations.
    """
    labels = df["LST"]
    classes = [labels_to_classes[round(label,2)] for label in labels]
    df["LST"] = classes
    return df

def class_to_label(df,classes_to_labels):
    """
    Converts numerical class representations in the "LST" column of the DataFrame back to their corresponding labels.

    Args:
    df (DataFrame): Input DataFrame.
    classes_to_labels (dict): Dictionary mapping classes to labels.

    Returns:
    DataFrame: DataFrame with numerical class representations converted to labels.
    """
    classes = df["LST"]
    labels = [classes_to_labels[clas] for clas in classes]
    df["LST"] = labels
    return df


def save_config(config):

    file_path = MODEL_CONFIG_REG
    # Step 3: Save the configuration dictionary to a JSON file
    with open(file_path, 'w') as f:
        json.dump(config, f, indent=4)  # `indent=4` makes the file readable

def load_config(classif):

    if classif:
        file_path = MODEL_CONFIG_CLASS

    else:
        file_path = MODEL_CONFIG_REG

    with open(file_path, 'r') as f:
        config = json.load(f)  # `indent=4` makes the file readable

    return config

def save_history(history,file):
    with open(file, "w") as f:
        json.dump(history,f, indent=4)


def scale_data(X_train:pd.DataFrame, X_test:pd.DataFrame, X_val:pd.DataFrame,scaler:StandardScaler):
    """
    Scale data using StandardScaler

    Args:
    X_train (array-like): Training data to be scaled.
    X_test (array-like): Testing data to be scaled.
    X_val (array-like): Validation data to be scaled.

    Returns:
    tuple: Scaled versions of X_train, X_test, and X_val
    """

    if not X_train.empty:
        X_train = scaler.fit_transform(X_train)
     
        # dump(scaler,MODEL_SCALER_PATH_REG)
    if not X_test.empty:
        X_test = scaler.transform(X_test)
    if not X_val.empty:
        X_val = scaler.transform(X_val)
    return X_train, X_test, X_val, scaler


def compute_history(history):
    """
    Compute the history for it to be ok to store
    """
    combined_history = {key: [] for key in history[0].keys()}
    st.session_state.stop_train = 0
    for h in history:
        for key in combined_history.keys():
            combined_history[key].extend(h[key])
    history = combined_history

    return history