import streamlit as st
import qgis.core
import qgis.gui
import qgis.utils
from qgis.PyQt.QtCore import QFileInfo
from qgis.core import QgsProject, QgsRasterLayer
from streamlit_folium import st_folium
import folium
import os

# Initialisation de QGIS
qgis.core.QgsApplication.setPrefixPath(os.environ['QGIS_PREFIX_PATH'], True)  # Remplacez par le chemin d'installation de QGIS
qgis_app = qgis.core.QgsApplication([], False)
qgis_app.initQgis()

# Titre de l'application
st.title("Visualisation de Températures Géographiques avec QGIS")

# Téléchargement des fichiers TIF
uploaded_files = st.file_uploader("Téléchargez vos fichiers TIF", type="tif", accept_multiple_files=True)

# Initialisation de la carte Folium
m = folium.Map(location=[0, 0], zoom_start=2)

# Chargement et gestion des couches
if uploaded_files:
    layers = []
    for uploaded_file in uploaded_files:
        # Sauvegarder le fichier uploadé temporairement
        with open(os.path.join("/tmp", uploaded_file.name), "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Charger le fichier TIF en tant que couche QGIS
        file_info = QFileInfo(os.path.join("/tmp", uploaded_file.name))
        layer_name = file_info.baseName()
        raster_layer = QgsRasterLayer(file_info.filePath(), layer_name)

        if not raster_layer.isValid():
            st.error(f"Le fichier {uploaded_file.name} n'est pas un raster valide.")
        else:
            QgsProject.instance().addMapLayer(raster_layer)
            layers.append(raster_layer)

    # Afficher les couches avec des cases à cocher
    for layer in layers:
        show_layer = st.checkbox(f"Afficher {layer.name()}", value=True)
        if show_layer:
            # Conversion de la couche QGIS en format compatible avec Folium (par exemple, GeoJSON, ou autre)
            extent = layer.extent()
            bounds = [[extent.yMinimum(), extent.xMinimum()], [extent.yMaximum(), extent.xMaximum()]]
            folium.raster_layers.ImageOverlay(
                name=layer.name(),
                image=os.path.join("/tmp", layer.name() + ".tif"),
                bounds=bounds,
                opacity=0.6,
            ).add_to(m)

    folium.LayerControl().add_to(m)
    st_folium(m, width=700, height=500)

# Terminer l'application QGIS
qgis_app.exitQgis()
