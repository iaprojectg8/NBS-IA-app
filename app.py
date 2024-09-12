import streamlit as st
import leafmap.foliumap as leafmap
from localtileserver import get_folium_tile_layer
import tempfile
import folium
from streamlit_folium import st_folium
import os
import rasterio
from localtileserver import get_leaflet_tile_layer



def display_rasters_on_map(raster_files):
    """
    Display the uploaded rasters on a map using leafmap.
    """
    try:
        # Créer une carte Leafmap
        st.title("Raster Map Viewer")
        m = leafmap.Map()

        for uploaded_file in raster_files:
            print(uploaded_file.name)
            # Enregistrer le fichier raster temporairement
            with tempfile.NamedTemporaryFile(delete=False, suffix=".tif") as temp_file:
                temp_file.write(uploaded_file.read())
                temp_file_path = temp_file.name

            # Ajouter le raster comme couche sur la carte
            if "LST" in uploaded_file.name:
                m.add_raster(temp_file_path, indexes=7, colormap='jet',layer_name=uploaded_file.name,opacity=1)
            else: 
                m.add_raster(temp_file_path, indexes=1, colormap='jet',layer_name=uploaded_file.name,opacity=1)

        # Afficher la carte dans Streamlit
        m.to_streamlit()

    except Exception as e:
        st.error(f"An error occurred: {e}")

# Interface utilisateur Streamlit
st.title("Upload Raster Files")
uploaded_files = st.file_uploader("Choose TIFF files", type=["tif", "tiff"], accept_multiple_files=True)

if uploaded_files:
    display_rasters_on_map(uploaded_files)