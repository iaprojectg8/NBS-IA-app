import streamlit as st
import leafmap.foliumap as leafmap
from localtileserver import get_folium_tile_layer
# import tempfile
import folium
from streamlit_folium import st_folium
import os

import os
from localtileserver import get_leaflet_tile_layer

def create_tile_layer(tiff_path):
    """
    Create a local tile server from a TIFF file.
    
    Args:
        tiff_path (str): Path to the TIFF file.
    """
    # Create a temporary directory for tiles
    temp_dir = "temp_tiles"
    os.makedirs(temp_dir, exist_ok=True)

    # Create the tile server
    tile_layer = get_leaflet_tile_layer(tiff_path)
    

    return tile_layer

# Example usage
tiff_path = "LST_32638.tif"
tile_dir = create_tile_layer(tiff_path)


def display_raster_on_map():
    """
    Display the uploaded raster on a map using leafmap.
    """
    try:
        # Create a leafmap map
        st.title("TEST page")
        m = leafmap.Map()


        # # Add the raster file directly as a layer on the map
        tiff_path = "LST_32638.tif"
        tile_layer = create_tile_layer(tiff_path,)

        m.add_layer(tile_layer, colormap='viridis', opacity=0.6)

        # Display the map in Streamlit
        m.to_streamlit()

    except Exception as e:
        st.error(f"An error occurred: {e}")

display_raster_on_map()