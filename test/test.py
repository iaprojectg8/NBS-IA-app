# ça c'est la base de tout mais ça marche

import matplotlib.pyplot as plt
import rasterio

# Charger le fichier raster
with rasterio.open("LST_32638.tif") as src:
    raster = src.read(1)  # Lire la première bande

# Afficher avec matplotlib
plt.imshow(raster, cmap='jet')
plt.colorbar()
plt.show()

