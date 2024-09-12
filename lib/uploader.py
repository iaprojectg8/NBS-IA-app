from utils.imports import *

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
    try:
        # Create a Leafmap map
        st.title("Raster Map Viewer")
        

        for uploaded_file in raster_files:
            # Save the raster file temporarily
            if uploaded_file.type=="image/tiff":
                
                # Display the map in Streamlit
                raster_vis(uploaded_file=uploaded_file)

            elif uploaded_file.type=="application/vnd.ms-excel":
                
                df = manage_csv(uploaded_file=uploaded_file)
                
                
    except Exception as e:
        st.error(f"An error occurred: {e}")



def raster_vis(uploaded_file):
    m = leafmap.Map()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".tif") as temp_file:
        temp_file.write(uploaded_file.read())
        temp_file_path = temp_file.name

    # Add the raster as a layer on the map
    if "LST" in uploaded_file.name:
        m.add_raster(temp_file_path, indexes=7, colormap='jet', layer_name=uploaded_file.name, opacity=1)
    else:
        m.add_raster(temp_file_path, indexes=1, colormap='jet', layer_name=uploaded_file.name, opacity=1)
    m.to_streamlit()


def manage_csv(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as temp_file:
        temp_file.write(uploaded_file.read())
        temp_file_path = temp_file.name
    df = pd.read_csv(temp_file_path)

    st.dataframe(df)
    return df