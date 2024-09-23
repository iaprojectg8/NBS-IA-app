from utils.imports import *
from lib.visualization import basic_visualization
from lib.callbacks import callback_exit, callback_save

def rdf_regressor(X, y, estimator, test_size):
    """
    Performs a random forest regression.

    Parameters:
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
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        st.session_state.scaler = scaler

        st.write("Training the Random Forest model...")
        rf_model = RandomForestRegressor(n_estimators=estimator, random_state=42,n_jobs=-1,verbose=1)
        st.session_state.estimator = estimator
        rf_model.fit(X_train, y_train)

        st.write("Making graphs")
    basic_visualization(X_test=X_test, y_test=y_test, model = rf_model)
    
    # Put the scaler with the model to unpack it with the model at the wanted moment.
    model_scaler_dict = {
        'model': rf_model,
        'scaler': scaler
    }
    st.session_state.model_scaler_dict = model_scaler_dict
    
    return model_scaler_dict



def display_all_precedent_training_graphs():
    st.plotly_chart(st.session_state.stat_on_pred_fig1)
    st.plotly_chart(st.session_state.stat_on_pred_fig2)
    st.plotly_chart(st.session_state.results_fig1)
    st.plotly_chart(st.session_state.results_fig2)


def save_model():
    """
    Update the model path and display the button to save the model
    """
    
    st.text_input("Folder path", key='input_path',value=st.session_state.input_path)
    st.button("Save model", on_click=callback_save)
    st.button("Exit", on_click=callback_exit)
    
