This repo contains the code for "Impact of Simulated Climate Data on Wind Power Prediction and Long-term Grid Planning", presented at NAPS 2024.
## Getting started
The required packages can be installed and activated by:
```
conda env create -f environment.yml
conda activate climate4grid
```
Section 2 covers the wind speed analysis. It provides a detailed probability distribution analysis and compares the use of climate data for historical generation through wind speed profiles against historical datasets. Additionally, it includes a seasonal comparison of both high-resolution (HR) and low-resolution (LR) datasets with the historical data.

Section 3 contains the model to train an MLP to predict the generation. 
