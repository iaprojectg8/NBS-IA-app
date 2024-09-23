from utils.imports import *
from lib.preprocessing import *
from lib.visualization import basic_visualization
from utils.variables import SAVE_PATH
from lib.callbacks import *
from lib.raster_vis import get_min_max, create_grid
from utils.variables import RESULT_FOLDER



def test(X,X_train, model:RandomForestRegressor):


    y_pred = model.predict(X)
    y_train_pred = model.predict(X_train)
    
    return y_pred, y_train_pred

def create_raster(df, variable, map):
    filename = f"{variable}.tif"
    lat_min, lat_max, lon_min, lon_max = get_min_max(df)
    grid_values, pixel_size = create_grid(df,variable=variable)
    vmin = df["LST"].min()
    vmax = df["LST"].max()
    print(vmin, vmax)
    print(df["LST_pred"].min(), df["LST_pred"].max())
    print(df.describe())
    # Définir la taille des pixels et l'origine

    transform = from_origin(lon_min, lat_max, pixel_size, pixel_size)

    if not os.path.exists(RESULT_FOLDER):
        os.makedirs(RESULT_FOLDER)
    else:
        print(f"Folder already exists: {RESULT_FOLDER}")
    complete_path = os.path.join(RESULT_FOLDER,filename)
    if variable == "LST":

        with rasterio.open(complete_path, 'w', driver='GTiff', height=grid_values.shape[0],
                width=grid_values.shape[1], count=7, dtype=grid_values.dtype,
                crs='EPSG:4326', transform=transform) as dst:
            dst.write(grid_values, 7)
        st.success("TIF file done")
        map.add_raster(complete_path, indexes=7, colormap='jet', layer_name=variable, opacity=1)
    elif variable == "Diff_LST":
        print("here i am building the prediction image")
        with rasterio.open(complete_path, 'w', driver='GTiff', height=grid_values.shape[0],
                width=grid_values.shape[1], count=1, dtype=grid_values.dtype,
                crs='EPSG:4326', transform=transform) as dst:
            dst.write(grid_values, 1)
        st.success("TIF file done")
        map.add_raster(complete_path, indexes=1, colormap='jet', layer_name=variable, opacity=1, vmin=-10,vmax=10)    
    else :
        print("here i am building the prediction image")
        with rasterio.open(complete_path, 'w', driver='GTiff', height=grid_values.shape[0],
                width=grid_values.shape[1], count=1, dtype=grid_values.dtype,
                crs='EPSG:4326', transform=transform) as dst:
            dst.write(grid_values, 1)
        st.success("TIF file done")
        map.add_raster(complete_path, indexes=1, colormap='jet', layer_name=variable, opacity=1, vmin=17,vmax=43)    

    return map

