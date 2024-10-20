from utils.imports import *
from utils.variables import SAVE_PATH
from lib.callbacks import *
from lib.raster_vis import get_min_max, create_grid
from utils.variables import RESULT_FOLDER


def init_test():
    """
    Reset the global variable test when arriving on the train page
    """
    st.session_state.train=0

def test(X,X_train, model:RandomForestRegressor):
    """
    Make prediction with the model and data provided

    Args:
        X (pd.Dataframe) : Dataframe of the future observation
        X_train (pd.Dataframe) : Dataframe of the current observation, so the same used in the training process
        model (RandomForestRegressor) : Model trained before in the process
    Retuns:
        y_pred (ndarray) : LST predicted by the model on the future observations
        y_train_pred (ndarray) : LST prediction by the model on the current observation
    """
    print("Make prediction on the future...")
    y_pred = model.predict(X)
    print("Make prediction on the current...")
    y_train_pred = model.predict(X_train)
    
    return y_pred, y_train_pred

def create_raster(df, variable, map:leafmap.folium):
    """
    Function to create a raster from a dataframe

    Args:
        df (pd.Dataframe) : Source Dataframe from which the data comes from
        variable (pd.Dataframe) : 
        map (leafmap) : Map on which we add the raster 
    Retuns:
        y_pred (ndarray) : 
        y_train_pred (ndarray) : LST prediction by the model on the current observation
    """
    filename = f"{variable}.tif"
    _, lat_max, lon_min, _ = get_min_max(df)
    grid_values, pixel_size_lat, pixel_size_lon = create_grid(df,variable=variable)
    transform = from_origin(lon_min, lat_max, pixel_size_lon, pixel_size_lat)

    # Verify that the result folder exists
    if not os.path.exists(RESULT_FOLDER):
        os.makedirs(RESULT_FOLDER)

    else:
        print(f"Folder already exists: {RESULT_FOLDER}")
    complete_path = os.path.join(RESULT_FOLDER,filename)
  
    min, max = write_raster(path=complete_path, grid_values=grid_values, transform=transform)
    map.add_raster(complete_path, indexes=1, colormap='jet', layer_name=variable, opacity=1, vmin=min,vmax=max)   

def write_raster(path, grid_values, transform):
    """
    Function to create a raster from a dataframe

    Args:
        path  (string) : The path to take to save the raster
        grid_values (ndarray) : containing the LST values to write the raster properly
        transform (Affine) : Object that defines how to convert from geographic coordinates to pixel indices in a raster image.
    Returns:
        min (float) : Minimum of temperature observed in the dataframe
        max (float) : Maximum of temperature observed in the dataframe
    """
    min = 0
    max = 0
    with rasterio.open(path, 'w', driver='GTiff', height=grid_values.shape[0],
                width=grid_values.shape[1], count=1, dtype=grid_values.dtype,
                crs='EPSG:4326', transform=transform) as destination:
            destination.write(grid_values, 1)
            min = np.nanmin(grid_values)
            max = np.nanmax(grid_values)
    st.success("TIF file done")
    return min, max