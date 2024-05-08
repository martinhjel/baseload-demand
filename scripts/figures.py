# %%
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pyperclip
import streamlit as st
from empire.input_client.client import EmpireInputClient
from empire.output_client.client import EmpireOutputClient
from IPython.display import display

from empire_profitability.analyse_baseload import get_generation_share, get_installed_capacity_share, get_parameters

# %%
result_folder = Path("../OpenEMPIRE") / "Results/genesis/1_node_baseload"
all_cases = [i for i in result_folder.glob("*") if i.name[0] != "."]

# %%

d = []
data = []
for c in sorted([i for i in all_cases if "7000" in i.name]):
    input_client = EmpireInputClient(c / "Input/Xlsx")
    output_client = EmpireOutputClient(c / "Output")
    df = input_client.nodes.get_electric_annual_demand()
    demand = df.loc[df["Nodes"] == "Node1", "ElectricAdjustment in MWh per hour"][0]

    df = output_client.get_node_operational_values()
    min_load = (-df.loc[df["Node"] == "Node1", "Load_MW"]).min()

    print(c, demand, min_load)
    d.append(-df["Load_MW"])
    data.append(go.Scatter(x=df["Hour"], y=(-df["Load_MW"]), name=c.name))

layout = go.Layout(template="plotly")
fig = go.Figure(data=data, layout=layout)
fig.show()

# %%


# %%% Base case

case1 = result_folder / "ncc_6000_co2_150_scale_1.0_shift0"

output_client = EmpireOutputClient(case1 / "Output")

df = output_client.get_node_operational_values()
df = df[df["Node"] == "Node1"]
df.loc[:, "Load_MW"] = -df["Load_MW"]

px.line(df, x="Hour", y="Load_MW")
df["Load_MW"].min()

# %% Check generation == load
case1 = result_folder / "ncc_5000_co2_150_scale_1.0_shift50"

output_client = EmpireOutputClient(case1 / "Output")

df_generation = output_client.get_europe_plot_generator_annual_production()

(df_generation.sum() / df_generation.sum().sum() * 100).round(2)

df = output_client.get_node_operational_values()
df["Load_MW"].sum().sum() / 1e3


# %%

case1 = result_folder / "ncc_5000_co2_150_scale_1.0_shift50"

output_client = EmpireOutputClient(case1 / "Output")
output_client.get_europe_plot_generator_installed_capacity()
output_client.get_europe_plot_generator_annual_production()
output_client.get_europe_summary_generator_types()

df = output_client.get_generators_values()
df_gas = df[df["GeneratorType"].isin(["OCGT", "CCGT"])]

gas_capacity_factor = (
    1e3 * df_gas["genExpectedAnnualProduction_GWh"].sum() / (df_gas["genInstalledCap_MW"].sum() * 8760)
)

# %% Gas capacity factor

case1 = result_folder / "ncc_5000_co2_150_scale_1.0_shift50"
output_client = EmpireOutputClient(case1 / "Output")
df = output_client.get_generators_values()
df_gas = df[df["GeneratorType"].isin(["OCGT", "CCGT"])]

gas_capacity_factor = (
    1e3 * df_gas["genExpectedAnnualProduction_GWh"].sum() / (df_gas["genInstalledCap_MW"].sum() * 8760)
)
print(f"{case1.name}: {gas_capacity_factor*100:.2f} %")

case1 = result_folder / "ncc_7000_co2_150_scale_1.0_shift50"
output_client = EmpireOutputClient(case1 / "Output")
df = output_client.get_generators_values()
df_gas = df[df["GeneratorType"].isin(["OCGT", "CCGT"])]

gas_capacity_factor = (
    1e3 * df_gas["genExpectedAnnualProduction_GWh"].sum() / (df_gas["genInstalledCap_MW"].sum() * 8760)
)
print(f"{case1.name}: {gas_capacity_factor*100:.2f} %")

# %% Absolute OCGT

case1 = result_folder / "ncc_5000_co2_150_scale_1.0_shift50"
output_client = EmpireOutputClient(case1 / "Output")
df = output_client.get_europe_plot_generator_installed_capacity()
print(case1.name)
display(df)

case1 = result_folder / "ncc_7000_co2_150_scale_1.0_shift50"
output_client = EmpireOutputClient(case1 / "Output")
df = output_client.get_europe_plot_generator_installed_capacity()
print(case1.name)
display(df)


# %%

nccs = []
for c in all_cases:
    ncc, co2, scale, shift = get_parameters(c)
    nccs.append(ncc)

nccs = np.unique(nccs)

capacity_mix = {}
generation_mix = {}
for ncc in nccs:
    st.markdown(f"## Nuclear capital cost: {ncc}")
    cases = []
    for c in all_cases:
        my_ncc, co2, scale, shift = get_parameters(c)
        if ncc == my_ncc:
            cases.append(c)

    df_installed = get_installed_capacity_share(cases)
    df_energy = get_generation_share(cases)

    capacity_mix[ncc] = df_installed
    generation_mix[ncc] = df_energy

# df = capacity_mix[ncc].T

# %% Capacity mix
colormap = px.colors.qualitative.Plotly

generators = ["Nuclear", "OCGT", "CCGT", "Solar", "Windonshore"]
symbol_map = {5000.0: "circle", 6000.0: "square", 7000.0: "star"}
opacity = {5000.0: 1, 6000.0: 0.8, 7000.0: 0.6}
linewidth = {5000.0: 5, 6000.0: 2.5, 7000.0: 1}

data = []
for j, ncc in enumerate(capacity_mix):
    df = capacity_mix[ncc].T
    for i, gen in enumerate(generators):
        data.append(
            go.Scatter(
                x=df.index,
                y=df[gen],
                name=f"{gen}-{ncc}",
                line=dict(color=colormap[i], width=linewidth[ncc]),
                mode="lines",
                opacity=opacity[ncc],
            )
        )  # , marker=dict(symbol=symbol_map[ncc])


layout = go.Layout(
    title=None,
    yaxis_title="Installed capacity ",
    xaxis_title="Change of baseload demand [MWh/h]",
    plot_bgcolor="white",  # White plot background
    paper_bgcolor="white",  # White figure background
    xaxis=dict(
        showline=True,  # Show x-axis line
        linecolor="black",  # x-axis line color
        mirror=True,
        gridcolor="lightgrey",  # x-axis grid line color
        ticks="inside",
    ),
    yaxis=dict(
        showline=True,  # Show y-axis line
        linecolor="black",  # y-axis line color
        mirror=True,
        gridcolor="lightgrey",  # y-axis grid line color
        ticks="inside",
    ),
    margin=dict(l=40, r=40, t=40, b=40),  # Adjust margins to fit the box
    autosize=False,  # Disable autosize to adjust dimensions
    width=760,  # Width of the figure
    height=400,  # Height of the figure
)
fig = go.Figure(data=data, layout=layout)

fig.show()
fig.write_image("images/load_and_ncc_changes_capacity_mix.pdf")


# %% Generation mix

data = []
for j, ncc in enumerate(generation_mix):
    df = generation_mix[ncc].T
    for i, gen in enumerate(generators):
        data.append(
            go.Scatter(
                x=df.index,
                y=df[gen],
                name=f"{gen}-{ncc}",
                line=dict(color=colormap[i], width=linewidth[ncc]),
                mode="lines",
                opacity=opacity[ncc],
            )
        )  # , marker=dict(symbol=symbol_map[ncc])


layout = go.Layout(
    title=None,
    yaxis_title="Installed generation share [%]",
    xaxis_title="Change of baseload demand [MWh/h]",
    plot_bgcolor="white",  # White plot background
    paper_bgcolor="white",  # White figure background
    xaxis=dict(
        showline=True,  # Show x-axis line
        linecolor="black",  # x-axis line color
        mirror=True,
        gridcolor="lightgrey",  # x-axis grid line color
        ticks="inside",
    ),
    yaxis=dict(
        showline=True,  # Show y-axis line
        linecolor="black",  # y-axis line color
        mirror=True,
        gridcolor="lightgrey",  # y-axis grid line color
        ticks="inside",
    ),
    margin=dict(l=40, r=40, t=40, b=40),  # Adjust margins to fit the box
    autosize=False,  # Disable autosize to adjust dimensions
    width=760,  # Width of the figure
    height=400,  # Height of the figure
)
fig = go.Figure(data=data, layout=layout)

fig.show()
fig.write_image("images/load_and_ncc_changes_generation_mix.pdf")

# %% Dispatch and load
case = all_cases[0]

input_client = EmpireInputClient(case / "Input/Xlsx")
output_client = EmpireOutputClient(case / "Output")

node = "Node1"
scenario = "scenario1"
period = "2020-2025"

df_operational_node_all = output_client.get_node_operational_values()
df_operational_node = df_operational_node_all.query(f"Node == '{node}'")

filtered_df = df_operational_node.query(f"Scenario == '{scenario}' and Period == '{period}'")

columns = [
    i for i in filtered_df.columns if "_MW" in i and i not in ["AllGen_MW", "Net_load_MW", "storEnergyLevel_MWh"]
]

current_columns = list(set(columns).intersection(set(filtered_df.columns)))

column_sums = filtered_df[current_columns].sum().abs()

# Find columns where the absolute sum is less than 0.01 MW
sum_hours = filtered_df["Hour"].max()
columns_to_drop = column_sums[column_sums < 0.01 * sum_hours].index
filtered_columns = set(current_columns).difference(set(columns_to_drop.to_list() + ["Load_MW"]))
filtered_columns = list(filtered_columns.union(set(["LoadShed_MW"])))  # Include if it was removed

# Melting the DataFrame to have a long-form DataFrame which is suitable for line plots in plotly
melted_df = pd.melt(filtered_df, id_vars=["Hour"], value_vars=filtered_columns)

# Calculate the sum of values for each variable
sums = melted_df.groupby("variable")["value"].sum()

# Sort variables based on their sums for a more readable area plot
sorted_variables = sums.sort_values(ascending=False).index.tolist()

# Sort the DataFrame based on the sorted order of variables
melted_df["variable"] = pd.Categorical(melted_df["variable"], categories=sorted_variables, ordered=True)
melted_df = melted_df.sort_values("variable")

# Creating the line plot
fig = px.area(
    melted_df,
    x="Hour",
    y="value",
    color="variable",
    # title=f"Operational values for {node}, {scenario}, {period}",
    template="plotly",
)
fig.add_trace(go.Scatter(x=filtered_df["Hour"], y=-filtered_df["Load_MW"], name="Load_MW", line=dict(width=1)))

fig.update_layout(
    template="plotly",
    xaxis=dict(
        title="Hour",
        domain=[0.3, 1],
        showline=True,  # Show x-axis line
        linecolor="black",  # x-axis line color
        mirror=True,
        gridcolor="lightgrey",  # x-axis grid line color),
    ),
    yaxis=dict(title="Value (MW)"),
    yaxis2=dict(
        title="Energy Price [EUR/MWh]",
        side="left",
        overlaying="y",
        showgrid=False,  # Hides the secondary y-axis gridlines if desired
        # tickmode="auto",  # Ensures ticks are automatically generated
        anchor="free",
        position=0.15,
        range=[0, 800],
    ),
    plot_bgcolor="white",  # White plot background
    paper_bgcolor="white",  # White figure background
    margin=dict(l=40, r=40, t=40, b=40),  # Adjust margins to fit the box
    autosize=False,  # Disable autosize to adjust dimensions
    width=1000,  # Width of the figure
    height=500,  # Height of the figure
)
fig.add_trace(
    go.Scatter(
        x=filtered_df["Hour"], y=filtered_df["Price_EURperMWh"], name="Energy Price", yaxis="y2", line=dict(width=1)
    )
)

fig.show()
fig.write_image("images/dispatch.pdf")


# %%% Plot duration curves


def get_duration_curve(df, y_axis_range=None):
    data = []
    for tech in df.columns:
        data.append(go.Scatter(x=[i for i in range(8760)], y=df[tech].sort_values(ascending=False), name=tech))
    layout = go.Layout(
        template="plotly",
        title="",
        yaxis_title="Load [MWh/h]",
        xaxis_title="Time [Hour]",
        plot_bgcolor="white",  # White plot background
        paper_bgcolor="white",  # White figure background
        xaxis=dict(
            showline=True,  # Show x-axis line
            linecolor="black",  # x-axis line color
            mirror=True,
            gridcolor="lightgrey",  # x-axis grid line color
            ticks="inside",
        ),
        yaxis=dict(
            showline=True,  # Show y-axis line
            linecolor="black",  # y-axis line color
            mirror=True,
            gridcolor="lightgrey",  # y-axis grid line color
            range=y_axis_range,
            ticks="inside",
        ),
        margin=dict(l=40, r=40, t=40, b=40),  # Adjust margins to fit the box
        autosize=False,  # Disable autosize to adjust dimensions
        width=760,  # Width of the figure
        height=400,  # Height of the figure
    )
    return go.Figure(data=data, layout=layout)


case1 = result_folder / "ncc_5000_co2_150_scale_1.0_shift50"


def get_df(case):
    output_client = EmpireOutputClient(case / "Output")
    df = output_client.get_node_operational_values()

    df = df[df["Node"] == "Node1"]
    df = df[
        [
            "Hour",
            "Load_MW",
            "Net_load_MW",
            "Nuclear_MW",
            "OCGT_MW",
            "CCGT_MW",
            "Solar_MW",
            "Windonshore_MW",
            "Windoffshore_MW",
            "Hydrorun-of-the-river_MW",
            "storCharge_MW",
            "storDischarge_MW",
            "LoadShed_MW",
        ]
    ]

    df.loc[:, "Load_MW"] = -df["Load_MW"]
    df.loc[:, "Net_load_MW"] = df["Load_MW"] - df["Windonshore_MW"] - df["Solar_MW"]
    df = df.set_index("Hour")
    df = df.loc[:, df.abs().sum() > 1e-4]
    return df


def get_curtailed(case):
    output_client = EmpireOutputClient(case / "Output")
    df_curtailed = output_client.get_curtailed_operational()
    df_curtailed = df_curtailed.pivot_table(
        index=["Node", "Period", "Scenario", "Season", "Hour"], columns=["RESGeneratorType"], values=["Curtailment_MWh"]
    )
    df_curtailed.index = df_curtailed.index.get_level_values("Hour")
    df_curtailed = df_curtailed.sort_index()
    df_curtailed.columns = df_curtailed.columns.get_level_values(1)
    df_curtailed = df_curtailed.drop(columns="Hydrorun-of-the-river")
    return df_curtailed


df = get_df(case1)
df_curtailed = get_curtailed(case1)
df_curtailed = df_curtailed.loc[:, df_curtailed.sum() > 1e-4]
for col in df_curtailed:
    df.loc[:, f"{col}_MW"] = df.loc[:, f"{col}_MW"] + df_curtailed[col]


fig1 = get_duration_curve(df, y_axis_range=[0, 160])
fig1.show()
fig1.write_image(f"images/duration_curve_{case1.name}.pdf")

case2 = result_folder / "ncc_7000_co2_150_scale_1.0_shift50"
df = get_df(case2)
df_curtailed = get_curtailed(case2)
df_curtailed = df_curtailed.loc[:, df_curtailed.sum() > 1e-4]
for col in df_curtailed:
    df.loc[:, f"{col}_MW"] = df.loc[:, f"{col}_MW"] + df_curtailed[col]

df["MyNetLoad"] = df["Load_MW"] - df["Solar_MW"] - df["Windonshore_MW"]

fig2 = get_duration_curve(df, y_axis_range=[-210, 330])
fig2.show()
fig2.write_image(f"images/duration_curve_{case2.name}.pdf")

# %% Land use

land_use_power = {
    "Nuclear": 587.17,
    "OCGT": 350.37,
    "CCGT": 350.37,
    "Solar": 9.13,
    "Windonshore": 1.49,
}

scenario_name = {5000: "Low", 6000: "Medium", 7000: "High"}


dispatchable_generators = ["OCGT", "CCGT", "Nuclear"]

data = []
summary = []
for shift in [-40, 0, 50, 100]:
    print(shift)
    base_load_cases = [i for i in all_cases if f"shift{shift}" in i.name]

    for case in base_load_cases:
        output_client = EmpireOutputClient(case / "Output")

        df = output_client.get_europe_summary_generator_types().copy(deep=True)
        df = df[df["GeneratorType"].isin(["Nuclear", "OCGT", "CCGT", "Solar", "Windonshore"])]
        df.drop(columns="Period")
        df = df.set_index(["GeneratorType"])
        df = df[["genInstalledCap_MW", "genExpectedAnnualProduction_GWh"]]

        dispatchable_mw = df.loc[df.index.isin(dispatchable_generators), "genInstalledCap_MW"].sum()
        total_mw = df.loc[:, "genInstalledCap_MW"].sum()

        df_land_use = pd.DataFrame(land_use_power, index=["Land use W/m2"]).T
        land_use = ((1 / df_land_use["Land use W/m2"]) * df["genInstalledCap_MW"]).sum()  # km2

        df = df[df > 1].dropna()
        df = (df / df.sum()) * 100
        df["NCC"] = case.name.split("_")[1]
        data.append(df)

        df_summary = output_client.get_europe_summary_emission_and_energy()
        df_summary = df_summary[
            ["AnnualCO2emission_Ton", "AnnualGeneration_GWh", "AvgCO2factor_TonPerMWh", "TotAnnualCurtailedRES_GWh"]
        ].copy(deep=True)
        df_summary["Capital Cost"] = int(case.name.split("_")[1])
        df_summary["Land Use [km$^2$]"] = land_use
        df_summary["Dispatchable generators"] = dispatchable_mw/total_mw
        df_summary.loc[:, "AvgCO2factor_TonPerMWh"] *= 1e3  # GWh
        # df_summary.loc[:,"TotAnnualCurtailedRES_GWh"] = df_summary.loc[:,"TotAnnualCurtailedRES_GWh"]/df_summary.loc[:,"AnnualGeneration_GWh"]
        df_summary["Shift"] = shift
        summary.append(df_summary)

df_summary = pd.concat(summary).sort_values(["Shift", "Capital Cost"])
df_summary.loc[:, "Capital Cost"] = df_summary["Capital Cost"].map(scenario_name)
df_summary = df_summary.set_index(["Shift", "Capital Cost"])
df_summary = df_summary[["AvgCO2factor_TonPerMWh", "TotAnnualCurtailedRES_GWh", "Land Use [km$^2$]", "Dispatchable generators"]]
df_summary = df_summary.rename(
    columns={"AvgCO2factor_TonPerMWh": "CO2 emissions", "TotAnnualCurtailedRES_GWh": "Curtailment"}
)


df_summary = ((df_summary / df_summary.loc[(0, "Medium")]) * 100).round(2)

for col in df_summary.columns:
    df_summary.loc[:, col] = df_summary[col].astype(str) + " \%"


pyperclip.copy(
    df_summary.to_latex(
        index=True,
        column_format="ll" + "".join(["r" for _ in df.columns]) + "",
        caption="Comparative analysis of CO2 emissions, curtailment, and land use with different baseload assumptions and nuclear energy costs. The presented values are percentwise relative to the medium cost scenario with no changes to the baseload. Land use estimates are based on median power density values sourced from \cite{Nland2022SpatialWorldwide}.",
        float_format="{:.2f}".format,
        label="tab:case-system",
    )
)

# %%
shift = 0
base_load_cases = [i for i in all_cases if f"shift{shift}" in i.name]

cases = [i for i in all_cases if f"shift{1}" in i.name and "7000" in i.name]
c = cases[0]
output_client = EmpireOutputClient(c / "Output")
df = output_client.get_europe_summary_generator_types().copy(deep=True)

print(c)
df[["GeneratorType", "genInvCap_MW"]]


# %% Storage

data = []
for case in base_load_cases:
    output_client = EmpireOutputClient(case / "Output")

    df_capacity = output_client.get_europe_plot_storage_installed_capacity()
    df_capacity.columns = pd.MultiIndex.from_product([[df_capacity.columns.name], df_capacity.columns])

    df_energy = output_client.get_europe_plot_storage_installed_energy()
    df_energy.columns = pd.MultiIndex.from_product([[df_energy.columns.name], df_energy.columns])

    df_storage = pd.concat([df_capacity, df_energy], axis=1)
    df_storage["Capital Cost"] = int(case.name.split("_")[1])
    data.append(df_storage)

df_storage = pd.concat(data)
df_storage

# %% Create dummy figure
fig = px.line(
    pd.DataFrame({"Hour": np.linspace(1, 24, 24, dtype=np.int32), "Demand": np.zeros(24) + 10}),
    x="Hour",
    y="Demand",
    template="plotly",
)

layout = go.Layout(
    title="",
    yaxis_title="Load [MWh/h]",
    xaxis_title="Time [Hour]",
    plot_bgcolor="white",  # White plot background
    paper_bgcolor="white",  # White figure background
    xaxis=dict(
        showline=True,  # Show x-axis line
        linecolor="black",  # x-axis line color
        mirror=True,
        gridcolor="lightgrey",  # x-axis grid line color
        ticks="inside",
    ),
    yaxis=dict(
        showline=True,  # Show y-axis line
        linecolor="black",  # y-axis line color
        mirror=True,
        gridcolor="lightgrey",  # y-axis grid line color
        range=[60, 160],
        ticks="inside",
    ),
    margin=dict(l=40, r=40, t=40, b=40),  # Adjust margins to fit the box
    autosize=False,  # Disable autosize to adjust dimensions
    width=600,  # Width of the figure
    height=300,  # Height of the figure
    template="plotly",
)
fig.update_layout(layout)
fig.show()
fig.write_image("images/load_template_figure.pdf")

# %%
