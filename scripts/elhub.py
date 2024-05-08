# %%
from functools import lru_cache

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests


class ElhubClient:
    """
    Client for fetching data from ElHub.

    Open datasets elhub: https://elhub.no/data/apnedata/#
    Overview consumption codes: https://dok.elhub.no/ediel2/Retningslinjer-for-n%C3%A6rings--og-forbrukskode.797376513.html
    """

    def __init__(self, base_url="https://api.elhub.no/energy-data/v0"):
        self.base_url = base_url

    def get_consumption_groups(self):
        url = f"{self.base_url}/consumption-groups/"
        response = requests.get(url)

        if response.status_code == 200:
            df_cs = pd.DataFrame.from_dict(response.json()["data"])
            df_cs = pd.concat([df_cs, df_cs["attributes"].apply(lambda x: pd.Series(x))], axis=1)
            df_cs = df_cs.drop(columns=["attributes"])
            return df_cs
        else:
            print(f"Failed to retrieve data: {response.status_code}")
            return pd.DataFrame()

    @lru_cache(maxsize=1)
    def get_consumption_data(self):
        url = "https://data.elhub.no/download/consumption_per_group_mba_hour/consumption_per_group_mba_hour-all-en-0000-00-00.csv"
        df = pd.read_csv(url, delimiter=",")
        df.loc[:, "START_TIME"] = pd.to_datetime(df["START_TIME"])
        df.loc[:, "END_TIME"] = pd.to_datetime(df["END_TIME"])
        df = df.set_index(["START_TIME", "END_TIME", "PRICE_AREA", "CONSUMPTION_GROUP"])
        return df


# Usage
elhub_client = ElhubClient()

# Replace '12345' with the actual consumption group ID
df_cs = elhub_client.get_consumption_groups()

# %%
df = elhub_client.get_consumption_data()

df = df.groupby(["START_TIME", "CONSUMPTION_GROUP"])[["VOLUME_KWH"]].sum()
df = df.reset_index("CONSUMPTION_GROUP").pivot(columns="CONSUMPTION_GROUP", values="VOLUME_KWH")
df.index = pd.to_datetime(df.index, utc=True)
df.index = df.index.tz_convert("Europe/Oslo")
df /= 1e6

# Remove outliers
df.loc["2023-12-20 20:00:00+01:00":"2023-12-21 06:00:00+01:00", "Tertiary Service"] = df.loc[
    "2023-12-20 20:00:00+01:00":"2023-12-21 06:00:00+01:00", "Tertiary Service"
].mask(df["Tertiary Service"] > 4)

# Fill with interpolate
df = df.interpolate()
# df = df.resample("1D").mean()

df["Year"] = df.index.year
df["Date and Time"] = df.index.map(lambda x: x.strftime("%d. %b %H:%M"))
df["Month"] = df.index.map(lambda x: x.strftime("%m"))
df["Day"] = df.index.dayofyear


df = df.sort_index()
colormap = px.colors.qualitative.Plotly

fig = go.Figure()
for year in df["Year"].unique():
    for i, col in enumerate(df.columns[:-4]):
        df_year = df[df["Year"] == year]
        if year == 2023:
            fig.add_trace(
                go.Scatter(
                    x=[i for i in range(len(df_year))],
                    y=df_year[col],
                    mode="lines",
                    name=f"{col}",
                    line=dict(color=colormap[i], width=1),
                    # legendgroup=col,
                    # legendgrouptitle_text=col,
                )
            )
        else:
            fig.add_trace(
                go.Scatter(
                    x=[i for i in range(len(df_year))],
                    y=df_year[col],
                    mode="lines",
                    line=dict(color=colormap[i], width=1),
                    showlegend=False,
                    # legendgroup=col,
                    # legendgrouptitle_text=col,
                )
            )

fig.update_layout(
    title=None,
    xaxis_title="Hour of the Year",
    yaxis_title="Load [GWh/h]",
    plot_bgcolor="white",  # White plot background
    paper_bgcolor="white",  # White figure background
    xaxis=dict(
        showline=True,  # Show x-axis line
        linecolor="black",  # x-axis line color
        mirror=True,
        gridcolor="lightgrey",  # x-axis grid line color
    ),
    yaxis=dict(
        showline=True,  # Show y-axis line
        linecolor="black",  # y-axis line color
        mirror=True,
        gridcolor="lightgrey",  # y-axis grid line color
    ),
    margin=dict(l=40, r=40, t=40, b=40),  # Adjust margins to fit the box
    autosize=False,  # Disable autosize to adjust dimensions
    width=1000,  # Width of the figure
    height=500,  # Height of the figure
)

fig.write_image("notebooks/images/load_norway.pdf")
