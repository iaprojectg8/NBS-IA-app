from utils.imports import *

def display_rasters_on_map(raster_files):
    """
    Display the uploaded rasters on a map using leafmap.
    
    Args:
        raster_files (list): List of uploaded raster files.
    """
    try:
        # Create a Leafmap map
        st.title("Raster Map Viewer")
        m = leafmap.Map()

        for uploaded_file in raster_files:
            # Save the raster file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=".tif") as temp_file:
                temp_file.write(uploaded_file.read())
                temp_file_path = temp_file.name

            # Add the raster as a layer on the map
            if "LST" in uploaded_file.name:
                m.add_raster(temp_file_path, indexes=7, colormap='jet', layer_name=uploaded_file.name, opacity=1)
            else:
                m.add_raster(temp_file_path, indexes=1, colormap='jet', layer_name=uploaded_file.name, opacity=1)

        # Display the map in Streamlit
        m.to_streamlit()

    except Exception as e:
        st.error(f"An error occurred: {e}")