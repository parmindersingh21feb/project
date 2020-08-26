# %% read csv file
import pandas as pd

df = pd.read_csv("2012.csv")


# %% make sure numeric columns have numbers, remove rows that are not
cols = [
    "perobs",
    "perstop",
    "age",
    "weight",
    "ht_feet",
    "ht_inch",
    "datestop",
    "timestop",
    "xcoord",
    "ycoord",
]

for col in cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna()


# %% make datetime column
df["datestop"] = df["datestop"].astype(str).str.zfill(8)
df["timestop"] = df["timestop"].astype(str).str.zfill(4)

from datetime import datetime


def make_datetime(datestop, timestop):
    year = int(datestop[-4:])
    month = int(datestop[:2])
    day = int(datestop[2:4])

    hour = int(timestop[:2])
    minute = int(timestop[2:])

    return datetime(year, month, day, hour, minute)


df["datetime"] = df.apply(lambda r: make_datetime(r["datestop"], r["timestop"]), axis=1)


# %% convert all value to label in the dataframe, remove rows that cannot be mapped
import numpy as np
from tqdm import tqdm

value_label = pd.read_excel(
    "2012 SQF File Spec.xlsx", sheet_name="Value Labels", skiprows=range(4)
)
value_label["Field Name"] = value_label["Field Name"].fillna(method="ffill")
value_label["Field Name"] = value_label["Field Name"].str.lower()
value_label["Value"] = value_label["Value"].fillna(" ")
value_label = value_label.groupby("Field Name").apply(
    lambda x: dict([(row["Value"], row["Label"]) for row in x.to_dict("records")])
)

cols = [col for col in df.columns if col in value_label]

for col in tqdm(cols):
    df[col] = df[col].apply(lambda val: value_label[col].get(val, np.nan))

df["trhsloc"] = df["trhsloc"].fillna("P (unknown)")
df = df.dropna()


#%% convert xcoord and ycoord to (lon, lot)
import pyproj

srs = "+proj=lcc +lat_1=41.03333333333333 +lat_2=40.66666666666666 +lat_0=40.16666666666666 +lon_0=-74 +x_0=300000.0000000001 +y_0=0 +ellps=GRS80 +datum=NAD83 to_meter=0.3048006096012192 +no_defs"
p = pyproj.Proj(srs)

df["coord"] = df.apply(lambda r: p(r["xcoord"], r["ycoord"], inverse=True), axis=1)

# %% convert height in feet/inch to cm
df["height"] = (df["ht_feet"] * 12 + df["ht_inch"]) * 2.54


#%% remove outlier
df = df[(df["age"] <= 100) & (df["age"] >= 10)]
df = df[(df["weight"] <= 350) & (df["weight"] >= 50)]


# %% delete columns that are not needed
df = df.drop(
    columns=[
        # processed columns
        "datestop",
        "timestop",
        "ht_feet",
        "ht_inch",
        "xcoord",
        "ycoord",
        # not useful
        "year",
        "recstat",
        "crimsusp",
        "dob",
        "ser_num",
        "arstoffn",
        "sumoffen",
        "compyear",
        "comppct",
        "othfeatr",
        "adtlrept",
        "dettypcm",
        "linecm",
        "repcmd",
        "revcmd",
        # location of stop
        # only use coord and city
        "addrtyp",
        "rescode",
        "premtype",
        "premname",
        "addrnum",
        "stname",
        "stinter",
        "crossst",
        "aptnum",
        "state",
        "zip",
        "addrpct",
        "sector",
        "beat",
        "post",
    ]
)


# %%
import folium
import seaborn as sns
import matplotlib.pyplot as plt


ax = sns.countplot(df["datetime"].dt.month)
ax.set(xlabel="Month of Year")


# %%
ax = sns.countplot(df["datetime"].dt.weekday)
# The day of the week with Monday=0, Sunday=6
ax.set_xticklabels(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
ax.set(xlabel="Day of Week")


# %%
ax = sns.countplot(df["datetime"].dt.hour)
ax.set(xlabel="Hour of Day")


# %%
ax = sns.countplot(data=df, x="race")
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, fontsize=5)
for p in ax.patches:
    percentage = p.get_height() * 100 / df.shape[0]
    txt = f"{percentage:.1f}%"
    x_loc = p.get_x()
    y_loc = p.get_y() + p.get_height()
    ax.text(x_loc, y_loc, txt)


# %%
ax = sns.countplot(df["race"], hue=df["sex"])
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, fontsize=5)


# %%
ax = sns.countplot(df["race"], hue=df["city"])
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, fontsize=5)


# %%
nyc = (40.730618, -73.935242)

m = folium.Map(location=nyc)


# %%
for coord in df.loc[df["detailcm"] == "ARSON", "coord"]:
    folium.CircleMarker(location=(coord[1], coord[0]), radius=1, color="red").add_to(m)

m


# %%
for coord in df.loc[df["detailcm"] == "KIDNAPPING", "coord"]:
    folium.CircleMarker(location=(coord[1], coord[0]), radius=1, color="yellow").add_to(
        m
    )

m

# %%
df.to_pickle("sqf.pkl")


# %%
forces = [col for col in df.columns if col.startswith("pf_")]
reasons_stop = [col for col in df.columns if col.startswith("cs_")]
result = ["arstmade", "sumissue", "frisked", "searched"]
df_selected = df[forces + result]

import seaborn as sns
import matplotlib.pyplot as plt

sns.heatmap((df_selected == "YES").astype(int).corr())

# %%
# reason for frisk and searched
reasons_frisk = [col for col in df.columns if col.startswith("rf_")]
is_frisked = df["searched"] == "YES"


# %%
ax = sns.countplot(x="searched", data=df[df["rf_vcrim"] == "YES"])
ax.set(title="rf_vcrim")

# %%
ax = sns.countplot(x="searched", hue="rf_vcrim", data=df)
for p in ax.patches:
    percentage = p.get_height() * 100 / df.shape[0]
    txt = f"{percentage:.1f}%"
    x_loc = p.get_x()
    y_loc = p.get_y() + p.get_height()
    ax.text(x_loc, y_loc, txt)

# %%
ax = sns.countplot(x="searched", hue="rf_verbl", data=df)
for p in ax.patches:
    percentage = p.get_height() * 100 / df.shape[0]
    txt = f"{percentage:.1f}%"
    x_loc = p.get_x()
    y_loc = p.get_y() + p.get_height()
    ax.text(x_loc, y_loc, txt)


# %%
df["race"].value_counts()

# %%
is_minority = ["ASIAN/PACIFIC ISLANDER", "OTHER", "AMERICAN INDIAN/ALASKAN NATIVE"]

df.loc[df["race"].isin(is_minority), "race"] = "MINORITY"


# %%
is_frisked = df["frisked"] == "YES"
in_selected_races = df["race"].isin(["BLACK", "WHITE"])

df_frisked_selected = df[is_frisked & in_selected_races]


ax = sns.countplot(data=df_frisked_selected, x="race", hue="city")

ax.set_xticklabels(ax.get_xticklabels(), rotation=45, fontsize=5)
for p in ax.patches:
    percentage = p.get_height() * 100 / df.shape[0]
    txt = f"{percentage:.1f}%"
    x_loc = p.get_x()
    y_loc = p.get_y() + p.get_height()
    ax.text(x_loc, y_loc, txt)

# %%

