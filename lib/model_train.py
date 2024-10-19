from utils.imports import *
from lib.visualization import basic_visualization
from lib.callbacks import callback_exit, callback_save
from utils.variables import MODEL_FILE_RDF, MODEL_FILE_NN, MODEL_FILE_SCALER


def init_train():
    """
    Reset the global variable test when arriving on the train page
    """
    st.session_state.test = 0

def train_rdf_regressor(X, y, estimator, test_size):
    """
    Performs a random forest regression.

    Args:
        df (pd.DataFrame): The dataframe containing the dataset to be used for rdf regression.
        parameters_list (list): A list containing the parameters to include and exclude in the model.
        estimator (int) : An integer that corresponds to the amount of estimators wanted for the forest

    Returns:
        None. The function outputs visualizations of the results after applying rdf regression.
    """
    with st.status("Model Training...",expanded=True):
        # Split the dataset
        st.write("Splitting the dataset...")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size , random_state=42)
        
        # Fit the scaler to your data and transform the data
        st.write("Scaling explainable variables...")
        scaler = StandardScaler()

        # Here i keep my data in a dataframe to be able to access to the variable names from the model
        X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
        X_test = scaler.transform(X_test)

        st.write("Training the Random Forest model...")
        rf_model = RandomForestRegressor(n_estimators=estimator, random_state=42,n_jobs=-1,verbose=1)
        st.session_state.estimator = estimator
        rf_model.fit(X_train, y_train)
        print("Model variable just after the fit",rf_model.feature_names_in_)

        st.write("Making graphs")

    # Visualization of the result graphs    
    basic_visualization(X_test=X_test, y_test=y_test, model = rf_model)
    return rf_model, scaler
    
    

def display_all_precedent_training_graphs():
    """
    Displays the graphs even after the trains to keep a record
    """

    if all(value is not None for value in [st.session_state.loss_history, st.session_state.mae_history, st.session_state.lr_history]):
    # Do something
        st.plotly_chart(st.session_state.loss_history)
        st.plotly_chart(st.session_state.mae_history) 
        st.plotly_chart(st.session_state.lr_history)

    st.plotly_chart(st.session_state.stat_on_pred_fig1)
    st.plotly_chart(st.session_state.stat_on_pred_fig2)
    st.plotly_chart(st.session_state.results_fig1)
    st.plotly_chart(st.session_state.results_fig2)

    

def model_save(model_chosen:str):
    """
    Update the model path and display the button to save the model
    """
    working_dir = os.getcwd()
    model_path = os.path.join(working_dir, "model")
         
    if "forest" in model_chosen.lower():
        
        st.session_state.model_path = os.path.join(model_path, MODEL_FILE_RDF)
        st.session_state.model_saving_config = st.session_state.model_scaler_dict 
        
        st.text_input("Model path", key='input_path',value=st.session_state.model_path)
        st.button("Save model", on_click=callback_save)
        st.button("Exit", on_click=callback_exit)
    else:
        st.session_state.model_path = os.path.join(model_path, MODEL_FILE_NN)
        st.session_state.scaler_path = os.path.join(model_path, MODEL_FILE_SCALER)
        st.session_state.model_saving_config = copy(st.session_state.model_scaler_dict)
        st.session_state.model_saving_config.pop("model")
        st.text_input("Model path", key='input_path',value=st.session_state.model_path)
        st.text_input("Scaler path", value = st.session_state.scaler_path)
        st.button("Save model", on_click=callback_save)
        st.button("Exit", on_click=callback_exit)

