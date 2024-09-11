
import folium
# import rasterio
# from rasterio.plot import show
import numpy as np
# import leafmap.foliumap as leafmap

import tempfile
import os
import matplotlib.pyplot as plt


# def file_uploader():
#     """
#     Function that initializes the expander in which the first map will be.
#     Args: 
#         json_gdf (dict): JSON object in which there are the coordinates of an uploaded shape file .
#     Returns:
#         output (dict) : It contains either the shape file coordiantes either the drawn shape ones
#     """
#     # Manage the file uploader
#     uploaded_files = st.file_uploader("Choose a tif file", type=["tif", "tiff"], accept_multiple_files=True)
#     if uploaded_files:

#         st.write("Files uploaded successfully!")
#         st.write(len(uploaded_files))

#         # Load the shapefile
#         load_shapefile(uploaded_files)


# def load_shapefile(uploaded_files):
#     """
#     Function to load a shape file and all its components, it needs at least the following list of 
#     file : shp, shx, prj.

#     Args:
#         uploaded_files (list) : List of uploaded shape files for a geometry

#     Returns:
#         gdf (geopandas.GeoDataFrame): Geodataframe containing the geometry of the files
#     """
#     # Create a temporary directory
#     with tempfile.TemporaryDirectory() as tmpdirname:
#         # Save the uploaded files to the temporary directory
#         for uploaded_file in uploaded_files:
        
#             file_path = os.path.join(tmpdirname, uploaded_file.name)
#             with open(file_path, 'wb') as f:
#                 f.write(uploaded_file.getbuffer())
        
       
#         tif_file = [os.path.join(tmpdirname, f.name) for f in uploaded_files if f.name.endswith('.tif')][0]

        
#         # Read the shapefile
#         with rasterio.open(tif_file) as src:
#             # Read the data as a NumPy array
#             data = src.read()

#             # Display metadata
#             print(src.meta)

#             # Plot the data (assuming it's a single-band raster)
#             show(data, cmap='gray')
#             plt.show()
       
    

# # Read a portion of the file to guess encoding
# with open("C:/Users/FlorianBERGERE/Keran/Groupe_Huit_Interne - Stage-IA/Dataset/Bangui/Data/LCZ.tif", 'rb') as f:
#     result = chardet.detect(f.read(1000))

# print(result)     
 
# with rasterio.open("C:/Users/FlorianBERGERE/Downloads/Tiff-Image-File-Download.tiff") as src:
#     # Read the image data
#     # Read the data
#     data = src.read(1)  # Read the first band

#     # Plot the data
#     plt.imshow(data, cmap='gray')
#     plt.colorbar()
#     plt.title('Raster Data')
#     plt.show()


# Print some information about the dataset
import matplotlib.pyplot as plt
import rasterio

# Charger le fichier raster
with rasterio.open("LST_32638.tif") as src:
    raster = src.read(7)  # Lire la première bande

# Afficher avec matplotlib
plt.imshow(raster, cmap='jet')
plt.colorbar()
plt.show()
