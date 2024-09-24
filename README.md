<div style="text-align: center;">
    <h1>Interactive App for Model Development and Raster Visualization</h1>
</div>


This application allows you to make model that are able to predict LST. You can visualize rasters, or variables in dataframe. Then you can also train and test your own model

## Features

- Visualize already created rasters
- Create and visualize rasters with data from dataframes
- Train Random Forest models on your dataframe
- See the result of the training process
- Save the model, to keep it on your computer
- Test your model on a future configuration of a city (with urban extension or NBS)
- See the difference that it makes on rasters

## Install
You should install miniconda to not have any problem with the installation as it will contain everything you need and well separate from anything else that could interfer. Interence between packages is the most annoying problem when making installation.

## Environment

If you don't have miniconda install it, and set it up correctly.

1. Create your conda environment
```
conda create --name env_name python=3.12
```
2. Acitvate it
```
conda activate env_name
```

3. Install the needed packages
```
conda install --file .\requirements.txt     
```

## Launch the app
```
streamlit run '.\Raster Visualization.py'
```