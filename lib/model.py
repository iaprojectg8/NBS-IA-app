from utils.imports import *
from preprocessing import *
from visualization import basic_visualization

def rdf_regressor(df,parameters_list, estimator):
    """
    Performs a random forest regression.

    Parameters:
        df (pd.DataFrame): The dataframe containing the dataset to be used for rdf regression.
        parameters_list (list): A list containing the parameters to include and exclude in the model.
        estimator (int) : An integer that corresponds to the amount of estimators wanted for the forest

    Returns:
        None. The function outputs visualizations of the results after applying rdf regression.
    """
    X,y = create_X_y(df, parameters_list) 

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2 , random_state=42)
    scaler = StandardScaler()

    # Fit the scaler to your data and transform the data
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    print("Training the model")
    rf_model = RandomForestRegressor(n_estimators=estimator, random_state=42,n_jobs=-1,verbose=1)
    rf_model.fit(X_train, y_train)
    
    # importance_vis(X_test=X_test, y_test=y_test, model=rf_model)
    basic_visualization(X_test=X_test, y_test=y_test, model = rf_model)



def adjusted_r2_calc(r2, X_test):
    """
    Calculate adjusted r2 knowing the r2 and X_test.

    Parameters:
        r2 (float): The r2 computed before
        X_test (pd.Dataframe): X_test is just here to give its dimension for the calculation 

    Returns:
        adjusted_r2 (float) : Output the results of adjusted R2 calculation.
    """
    n,k = X_test.shape
    print(k)
    print(n)
    adjusted_r2 = 1 - ((1 - r2) * (n - 1)) / ( n - k- 1)
    return adjusted_r2