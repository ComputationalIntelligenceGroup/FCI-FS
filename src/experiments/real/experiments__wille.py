# -*- coding: utf-8 -*-
from pathlib import Path
import sys

SRC_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SRC_DIR))

import numpy as np
import pandas as pd
from causaldiscovery.algorithms.FCI_SF import fci_sf
from causaldiscovery.CItest.noCache_CI_Test import myTest
from causallearn.graph.GeneralGraph import GeneralGraph 





path = Path(
    r"C:\Users\chdem\0UNIVERSIDAD\CIG\code\FCI-SF\src\experiments\real\13059_2004_896_MOESM1_ESM.txt"
)
# The first six lines are descriptive text.
raw = pd.read_csv(
    path,
    sep=r"\s+",
    skiprows=6,
)

# Gene-level annotations: do not use these directly as causal variables.
gene_metadata = raw[
    ["Pathwayname", "ECID", "AGI", "Genename", "Name", "Probeset"]
].copy()

# Extract c1, ..., c118.
expression = raw.filter(regex=r"^c\d+$").astype(float)

# Transpose: arrays become observations and genes become variables.
X = expression.T

# AGI identifiers are safer variable names than the informal gene names.
X.columns = gene_metadata["Genename"].str.upper().to_numpy()
X.index.name = "array"

print(raw.shape)  # (39, 124)
print(X.shape)    # (118, 39)

X_log = np.log2(X + 1)

X_standardized = (
    X_log - X_log.mean(axis=0)
) / X_log.std(axis=0, ddof=1)

assert X_standardized.shape == (118, 39)
assert not X_standardized.isna().any().any()

data = X_standardized.to_numpy()
variable_names = X_standardized.columns.tolist()

CI_test = myTest(X_standardized)
ALPHA = 0.05

fci_stable_full = fci_sf(data, independence_test_method=CI_test, initial_sep_sets = {}, alpha= ALPHA,  initial_graph = GeneralGraph([]), new_node_names = variable_names, verbose = False)
