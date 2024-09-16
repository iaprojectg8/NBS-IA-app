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

# # Example usage
# tiff_path = "LST_32638.tif"
# tile_dir = create_tile_layer(tiff_path)


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

# display_raster_on_map()


from math import ceil

def taille_echantillon(N, p, Z, E):
    """
    Calcule la taille d'un échantillon en fonction de la taille de la population,
    du taux de défaut attendu, du score Z et de la marge d'erreur acceptable.
    
    :param N: int, taille de la population
    :param p: float, taux de défaut (entre 0 et 1)
    :param Z: float, score Z pour le niveau de confiance souhaité (ex: 1.96 pour 95%)
    :param E: float, marge d'erreur (entre 0 et 1)
    :return: int, taille minimale de l'échantillon
    """
    numerator = N * (Z**2) * p * (1 - p)
    denominator = (N - 1) * (E**2) + (Z**2) * p * (1 - p)
    return ceil(numerator / denominator)

# # Exemple d'utilisation avec les paramètres donnés
# N = 6846  # Taille de la population
# p = 0.014  # Taux de défaut
# Z = 2.58  # Score Z pour un niveau de confiance de 95%
# E = 0.01  # Marge d'erreur de 1%

# taille = taille_echantillon(N, p, Z, E)
# print(f"Nombre de pièces à vérifier : {taille}")
import math


# Taux de pièces non défectueuses
p_non_def = 0.986  # 98.6%

# Niveau de confiance souhaité (1% de risque que toutes les pièces soient non défectueuses)
confiance = 0.02

# Résolution de l'équation pour trouver n
n = math.log(confiance) / math.log(p_non_def)
n = math.ceil(n)  # Nombre de pièces à vérifier (arrondi au nombre entier supérieur)

print(f"Nombre de pièces à vérifier pour détecter 100% des défauts avec 99% de confiance : {n}")



