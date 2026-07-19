import pandas as pd

path = r"C:\Users\chdem\Downloads\7681811\sachs\Data Files\\"

files = [
    "b2camp.csv",
    "cd3cd28.csv",
    "cd3cd28_icam2.csv",
    "cd3cd28_aktinhib.csv",
    "cd3cd28_g0076.csv",
    "cd3cd28_ly.csv",
    "cd3cd28_u0126.csv",
    "cd3cd28_psitect.csv",
    "pma.csv"
]

dfs = []
for fname in files:
    df = pd.read_csv(path + fname)
    df["condition"] = fname.replace(".csv", "")
    dfs.append(df)

combined = pd.concat(dfs, ignore_index=True)
print("Combined shape:", combined.shape)
print("Columns:", combined.columns)