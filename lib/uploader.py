from utils.imports import *
from utils.variables import REMAKE_FOLDER

def upload_files():
    """
    Handle file uploads from the user.
    Returns:
        List of uploaded files.
    """
    st.title("Upload Raster Files")
    uploaded_files = st.file_uploader("Choose TIFF or CSV files", type=["tif", "tiff", "csv"], accept_multiple_files=True)
    return uploaded_files


def manage_uploaded_files(raster_files):
    """
    Display the uploaded rasters on a map using leafmap.
    
    Args:
        raster_files (list): List of uploaded raster files.
    """
    
    # Create a Leafmap map
    st.title("Raster Map Viewer")
    
    map = leafmap.Map()
    for uploaded_file in raster_files:
        # Save the raster file temporarily
        if uploaded_file.type=="image/tiff":
           
            
            # Display the map in Streamlit
            raster_vis(uploaded_file=uploaded_file, map=map)

        elif uploaded_file.type=="application/vnd.ms-excel":
            
            df = manage_csv(uploaded_file=uploaded_file)
            variable_list = ["LST","LS1","LS2","LS3","LS4","LS5","LS6","OCCSOL","URB","ALT","EXP","PENTE","NATSOL","NATSOL2","HAUTA","CATHYD","ZONECL","ALB"]
            selected_variable = st.selectbox(label="Chose a variable to observe",options=variable_list)
            if selected_variable:
                create_and_display_raster(df, filename = f'{selected_variable}_remake.tif',map=map)
            

    map.to_streamlit()




def manage_csv(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as temp_file:
        temp_file.write(uploaded_file.read())
        temp_file_path = temp_file.name
    df = pd.read_csv(temp_file_path)

    st.dataframe(df)
    return df


def create_grid(df, variable):



    lat_unique = sorted(df['LAT'].unique())
    lon_unique = sorted(df['LON'].unique()) 
    pixel_size = np.mean(pd.DataFrame(lat_unique).diff())
    print(abs(pixel_size))
    print(len(lat_unique))
    grid_x, grid_y = np.meshgrid(lon_unique, lat_unique)
    # print(grid_x) 
    # print(grid_y)
    points = df[['LAT', 'LON']].values  # Coordonnées connues
    values = df[variable].values  # Valeurs associées (certaines sont NaN)
    grid_values = griddata(points, values, (grid_y, grid_x), method='nearest')
    grid_values = grid_values[::-1, :]
    print(grid_values)


    return grid_values, pixel_size



def get_min_max(df):
    lat_min= df['LAT'].min() 
    lat_max= df['LAT'].max()
    lon_min= df['LON'].min()
    lon_max= df['LON'].max()

    return lat_min, lat_max, lon_min, lon_max


def create_and_display_raster(df,filename, map):
    variable = filename.split("_")[0]
    lat_min, lat_max, lon_min, lon_max = get_min_max(df)
    grid_values, pixel_size = create_grid(df,variable=variable)
    # Définir la taille des pixels et l'origine

    transform = from_origin(lon_min, lat_max, pixel_size, pixel_size)

    # Check if the folder exists, if not, create it
    if not os.path.exists(REMAKE_FOLDER):
        os.makedirs(REMAKE_FOLDER)
    else:
        print(f"Folder already exists: {REMAKE_FOLDER}")
    complete_path = os.path.join(REMAKE_FOLDER,filename)
    if variable == "LST":
        with rasterio.open(complete_path, 'w', driver='GTiff', height=grid_values.shape[0],
                width=grid_values.shape[1], count=7, dtype=grid_values.dtype,
                crs='EPSG:4326', transform=transform) as dst:
            dst.write(grid_values, 7)
        st.success("TIF file done")
        map.add_raster(complete_path, indexes=7, colormap='jet', layer_name=variable, opacity=1)
    else :
        with rasterio.open(complete_path, 'w', driver='GTiff', height=grid_values.shape[0],
                width=grid_values.shape[1], count=1, dtype=grid_values.dtype,
                crs='EPSG:4326', transform=transform) as dst:
            dst.write(grid_values, 1)
        st.success("TIF file done")
        map.add_raster(complete_path, indexes=1, colormap='jet', layer_name=variable, opacity=1)    

    return map


def raster_vis(uploaded_file, map:leafmap):
    """
    Function to display the selected raster
    Args:
        - uploaded_file (UploadedFile) : object containing the file informations
        - map (leafmap) : map to display the raster on
    """

    with tempfile.NamedTemporaryFile(delete=False, suffix=".tif") as temp_file:
        temp_file.write(uploaded_file.read())
        temp_file_path = temp_file.name

    if "LST" in uploaded_file.name:
        map.add_raster(temp_file_path, indexes=7, colormap='jet', layer_name=uploaded_file.name, opacity=1)
    else:
        map.add_raster(temp_file_path, indexes=1, colormap='jet', layer_name=uploaded_file.name, opacity=1)
    
    return map