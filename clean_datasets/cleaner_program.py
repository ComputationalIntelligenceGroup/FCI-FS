# -*- coding: utf-8 -*-
"""
Created on Fri Nov 21 15:46:04 2025

@author: chdem
"""

import pandas as pd
from glob import glob

# Ruta a la carpeta con los CSV
ruta = r"../clean_datasets/real-world_datasets/*.csv"

# DataFrames list
dfs = [pd.read_csv(fichero) for fichero in glob(ruta)]



#df.to_csv("FRED-MD-Macroeconomic.csv")

