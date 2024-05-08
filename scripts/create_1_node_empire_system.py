from pathlib import Path

import numpy as np
import pandas as pd

from empire.input_client.client import EmpireInputClient
from empire.input_client.sheets_structure import sheets
from empire.input_client.utils import create_empty_empire_dataset

dataset_name = "1_node"
# periods = [1,2]

empire_path = Path.cwd() / "../OpenEMPIRE"
dataset_path = empire_path / f"Data handler/{dataset_name}"
existing_dataset_path = empire_path / "Data handler/test"

existing_input_client = EmpireInputClient(dataset_path=existing_dataset_path)

if not dataset_path.exists():
    create_empty_empire_dataset(dataset_path)
new_input_client = EmpireInputClient(dataset_path=dataset_path)

# Create excel files
for xls_file in sheets:
    file = dataset_path / f"{xls_file}.xlsx"
    file.parent.mkdir(exist_ok=True, parents=True)
    with pd.ExcelWriter(file, engine="openpyxl") as writer:
        for sheet in sheets[xls_file]:
            pd.DataFrame().to_excel(writer, sheet_name=sheet, index=False)


###########
# General #
###########
regular_seasons = ["winter", "summer", "fall", "spring"]
peak_seasons = ["peak1", "peak2"]

n_peak_hours = 24
n_regular_hours = int((8760 - n_peak_hours * len(peak_seasons)) / len(regular_seasons))

co2_cap = {"Period": [1], "CO2Cap [in Mton CO2eq]": [48321 / 1e6]}
co2_price = {"Period": [1], "CO2price in euro per tCO2": [0]}


def compute_season_scale_dataframe(regular_seasons, peak_seasons, n_regular_hours, n_peak_hours):
    n_regular_seasons = len(regular_seasons)
    n_peak_seasons = len(peak_seasons)

    regular_season_scale = (8760 - n_peak_seasons * n_peak_hours) / n_regular_seasons / n_regular_hours

    season_scale = {}
    for r_season in regular_seasons:
        season_scale[r_season] = regular_season_scale

    for p_season in peak_seasons:
        season_scale[p_season] = 1.0

    df_season_scale = pd.DataFrame(list(season_scale.items()), columns=["Season", "seasonScale"])

    return df_season_scale


df_season_scale = compute_season_scale_dataframe(
    regular_seasons=regular_seasons,
    peak_seasons=peak_seasons,
    n_regular_hours=n_regular_hours,
    n_peak_hours=n_peak_hours,
)

df_co2_cap = pd.DataFrame(co2_cap)
df_co2_price = pd.DataFrame(co2_price)

# Write data
new_input_client.general.set_co2_cap(df_co2_cap)
new_input_client.general.set_co2_price(df_co2_price)
new_input_client.general.set_season_scale(df_season_scale)


#############
# Generator #
#############

existing_input_client.generator.get_capital_costs().columns
columns = ["GeneratorTechnology", "Period", "generatorCapitalCost in euro per kW"]
capital_cost_data = [
    ["OCGT", 1, 320],
    ["CCGT", 1, 640],
    ["Solar", 1, 532],
    ["Windonshore", 1, 943],
    ["Windoffshore", 1, 1891],
    ["Hydro run-of-the-river", 1, 666e3],
    ["Nuclear", 1, 3600],
]

df_capital_costs = pd.DataFrame(capital_cost_data, columns=columns)

df_fixed_om_costs = existing_input_client.generator.get_fixed_om_costs()
columns = ["GeneratorTechnology", "Period", "generatorFixedOMCost in euro per kW"]
om_cost_data = [
    ["OCGT", 1, 15],
    ["CCGT", 1, 15],
    ["Solar", 1, 12],
    ["Windonshore", 1, 12],
    ["Windoffshore", 1, 26],
    ["Hydro run-of-the-river", 1, 666e3],
    ["Nuclear", 1, 120],
]
df_fixed_om_costs = pd.DataFrame(om_cost_data, columns=columns)

df_variable_om_costs = existing_input_client.generator.get_variable_om_costs()
columns = ["GeneratorTechnology", "generatorVariableOMcosts in euro per MWh"]
om_var_cost_data = [
    ["OCGT", 1.73],
    ["CCGT", 1.73],
    ["Solar", 0.0],
    ["Windonshore", 0.0],
    ["Windoffshore", 0.0],
    ["Hydro run-of-the-river", 0.0],
    ["Nuclear", 7.5],
]
df_variable_om_costs = pd.DataFrame(om_var_cost_data, columns=columns)

df_fuel_costs = existing_input_client.generator.get_fuel_costs()
columns = ["GeneratorTechnology", "Period", "generatorTypeFuelCost in euro per GJ"]
fuel_costs_data = [
    ["OCGT", 1, 48.5 / 3.6],  # 48.5 USD/MWh -> 48.5/3.6 EUR/GJ
    ["CCGT", 1, 48.5 / 3.6],
    ["Solar", 1, 0.0],
    ["Windonshore", 1, 0.0],
    ["Windoffshore", 1, 0.0],
    ["Hydro run-of-the-river", 1, 0.0],
    ["Nuclear", 1, 1.0404],
]
df_fuel_costs = pd.DataFrame(fuel_costs_data, columns=columns)

df_ccs_cost_ts_variable = existing_input_client.generator.get_ccs_cost_ts_variable()
columns = ["Period", "CCS_TScost in euro per tCO2"]
ccs_tscost = [[1, 14.0797035]]
df_ccs_cost_ts_variable = pd.DataFrame(ccs_tscost, columns=columns)

df_efficiency = existing_input_client.generator.get_efficiency()
columns = ["GeneratorTechnology", "Period", "generatorEfficiency"]
eff_data = [
    ["OCGT", 1, 0.39],
    ["CCGT", 1, 0.59],
    ["Solar", 1, 1.0],
    ["Windonshore", 1, 1.0],
    ["Windoffshore", 1, 1.0],
    ["Hydro run-of-the-river", 1, 1.0],
    ["Nuclear", 1, 1.0],
]
df_efficiency = pd.DataFrame(eff_data, columns=columns)

df_ref_initial_capacity = existing_input_client.generator.get_ref_initial_capacity()
columns = ["Node", "GeneratorTechnology", "generatoReferenceInitialCapacity in MW"]
init_cap_data = [
    ["Node1", "OCGT", 0.0],
    ["Node1", "CCGT", 0.0],
    ["Node1", "Solar", 0.0],
    ["Node1", "Windonshore", 0.0],
    ["Node1", "Windoffshore", 0.0],
    ["Node1", "Hydro run-of-the-river", 0.0],
    ["Node1", "Nuclear", 0.0],
]
df_ref_initial_capacity = pd.DataFrame(init_cap_data, columns=columns)

df_scale_factor_initial_capacity = existing_input_client.generator.get_scale_factor_initial_capacity()
columns = ["GeneratorTechnology", "Period", "generatorRetirementFactorInitialCap"]
scale_factor_initial_capacity_data = [
    ["OCGT", 1, 0.0],
    ["CCGT", 1, 0.0],
    ["Solar", 1, 0.0],
    ["Windonshore", 1, 0.0],
    ["Windoffshore", 1, 0.0],
    ["Hydro run-of-the-river", 1, 0.0],
    ["Nuclear", 1, 0.0],
]
df_scale_factor_initial_capacity = pd.DataFrame(scale_factor_initial_capacity_data, columns=columns)

df_initial_capacity = existing_input_client.generator.get_initial_capacity()
columns = ["Node", "GeneratorTechnology", "Period", "generatorInitialCapacity in MW"]
initial_capacity_data = [
    ["Node1", "OCGT", 1, 0.0],
    ["Node1", "CCGT", 1, 0.0],
    ["Node1", "Solar", 1, 0.0],
    ["Node1", "Windonshore", 1, 0.0],
    ["Node1", "Windoffshore", 1, 0.0],
    ["Node1", "Hydro run-of-the-river", 1, 0.0],
    ["Node1", "Nuclear", 1, 0.0],
]
df_initial_capacity = pd.DataFrame(initial_capacity_data, columns=columns)

df_max_built_capacity = existing_input_client.generator.get_max_built_capacity()
columns = ["Node", "GeneratorTechnology", "Period", "generatorMaxBuildCapacity in MW"]
max_built_cap_data = [
    ["Node1", "Gas", 1, 200e3],
]
df_max_built_capacity = pd.DataFrame(max_built_cap_data, columns=columns)

df_max_installed_capacity = existing_input_client.generator.get_max_installed_capacity()
columns = ["Node", "GeneratorTechnology", "generatorMaxInstallCapacity  in MW"]
max_installed_data = [
    ["Node1", "Gas", 200e3],
    ["Node1", "Wind_onshr", 200e3],
    ["Node1", "Wind_offshr", 200e3],
    ["Node1", "Solar", 200e3],
    ["Node1", "Nuclear", 200e3],
]
df_max_installed_capacity = pd.DataFrame(max_installed_data, columns=columns)

df_ramp_rate = existing_input_client.generator.get_ramp_rate()
columns = ["ThermalGenerators", "RampRate"]
ramp_rate_data = [["OCGT", 1], ["CCGT", 0.85], ["Nuclear", 0.3]]
df_ramp_rate = pd.DataFrame(ramp_rate_data, columns=columns)

df_generator_type_availability = existing_input_client.generator.get_generator_type_availability()
columns = ["Generator", "GeneratorTypeAvailability"]
type_availability_data = [
    ["OCGT", 1.0],
    ["CCGT", 1.0],
    ["Solar", 0.0],
    ["Windonshore", 0.0],
    ["Windoffshore", 0.0],
    ["Nuclear", 1.0],  # Replace by availability series. Requires changes to scenario_random.py
]
df_generator_type_availability = pd.DataFrame(type_availability_data, columns=columns)

df_co2_content = existing_input_client.generator.get_co2_content()
columns = ["GeneratorTechnology", "CO2Content_in_tCO2/GJ"]
co2_content_data = [["OCGT", 0.18 / 3.6], ["CCGT", 0.18 / 3.6], ["Nuclear", 0.0]]  # 0.18 MWh->GJ
df_co2_content = pd.DataFrame(co2_content_data, columns=columns)

df_lifetime = existing_input_client.generator.get_lifetime()
columns = ["GeneratorTechnology", "generatorLifetime"]
lifetime_data = [
    ["OCGT", 30],
    ["CCGT", 30],
    ["Solar", 25],
    ["Windonshore", 25],
    ["Windoffshore", 25],
    ["Hydro run-of-the-river", 40],
    ["HydroDummy1", 40],
    ["HydroDummy2", 40],
    ["Nuclear", 60],
]
df_lifetime = pd.DataFrame(lifetime_data, columns=columns)


# Write data
new_input_client.generator.set_capital_costs(df_capital_costs)
new_input_client.generator.set_fixed_om_costs(df_fixed_om_costs)
new_input_client.generator.set_variable_om_costs(df_variable_om_costs)
new_input_client.generator.set_fuel_costs(df_fuel_costs)
new_input_client.generator.set_ccs_cost_ts_variable(df_ccs_cost_ts_variable)
new_input_client.generator.set_efficiency(df_efficiency)
new_input_client.generator.set_ref_initial_capacity(df_ref_initial_capacity)
new_input_client.generator.set_scale_factor_initial_capacity(df_scale_factor_initial_capacity)
new_input_client.generator.set_initial_capacity(df_initial_capacity)
new_input_client.generator.set_max_built_capacity(df_max_built_capacity)
new_input_client.generator.set_max_installed_capacity(df_max_installed_capacity)
new_input_client.generator.set_ramp_rate(df_ramp_rate)
new_input_client.generator.set_generator_type_availability(df_generator_type_availability)
new_input_client.generator.set_co2_content(df_co2_content)
new_input_client.generator.set_lifetime(df_lifetime)


########
# SETS #
########

df_nodes = existing_input_client.sets.get_nodes()
columns = ["Node"]
nodes_data = [["Node1"], ["NodeDummy"]]
df_nodes = pd.DataFrame(nodes_data, columns=columns)

df_offshore_nodes = existing_input_client.sets.get_offshore_nodes()
columns = ["OffshoreNode"]
offshore_nodes_data = [["NodeDummy"]]
df_offshore_nodes = pd.DataFrame(offshore_nodes_data, columns=columns)

df_horizon = existing_input_client.sets.get_horizon()
columns = ["Horizon"]
horizon_data = [[1]]
df_horizon = pd.DataFrame(horizon_data, columns=columns)

df_storage = existing_input_client.sets.get_storage()
columns = ["Storage", "DependentStorage"]
storage_data = [["Hydro Pump Storage", "DummyStorage"]]
df_storage = pd.DataFrame(storage_data, columns=columns)

df_technology = existing_input_client.sets.get_technology()
columns = ["Technology"]
technology_data = [
    ["Gas"],
    ["Solar"],
    ["Wind_onshr"],
    ["Wind_offshr"],
    ["Hydro_reg"],
    ["Hydro_ror"],
    ["Nuclear"],
]
df_technology = pd.DataFrame(technology_data, columns=columns)

df_generators = existing_input_client.sets.get_generators()
generators_data = {
    "Generator": [
        "Nuclear",
        "OCGT",
        "CCGT",
        "Solar",
        "Windonshore",
        "Windoffshore",
        "Hydro run-of-the-river",
        "HydroDummy1",
        "HydroDummy2",
    ],
    "HydroGeneratorWithReservoir": ["HydroDummy1", "", "", "", "", "", "", "", ""],
    "HydroGenerator": ["HydroDummy2", "", "", "", "", "", "", "", ""],
    "ThermalGenerators": ["OCGT", "CCGT", "Nuclear", "", "", "", "", "", ""],
}
df_generators = pd.DataFrame(generators_data)

df_line_type = existing_input_client.sets.get_line_type()
columns = ["LineType"]
line_type_data = [["HVAC_OverheadLine"], ["HVDC_Cable"]]
df_line_type = pd.DataFrame(line_type_data, columns=columns)

df_storage_of_nodes = existing_input_client.sets.get_storage_of_nodes()
columns = ["Node", "Storage"]
storage_of_nodes_data = [["Node1", "Hydro Pump Storage"]]
df_storage_of_nodes = pd.DataFrame(storage_of_nodes_data, columns=columns)

df_directional_lines = existing_input_client.sets.get_directional_lines()
columns = ["NodeFrom", "NodeTo"]
directional_lines_data = [["Node1", "NodeDummy"], ["NodeDummy", "Node1"]]
df_directional_lines = pd.DataFrame(directional_lines_data, columns=columns)

df_line_type_of_directional_lines = existing_input_client.sets.get_line_type_of_directional_lines()
columns = ["FromNode", "ToNode", "LineType"]
line_type_of_directional_lines_data = [["Node1", "NodeDummy", "HVDC_Cable"], ["NodeDummy", "Node1", "HVDC_Cable"]]
df_line_type_of_directional_lines = pd.DataFrame(line_type_of_directional_lines_data, columns=columns)

df_generators_of_node = existing_input_client.sets.get_generators_of_node()
columns = ["Node", "Generator"]
generators_of_node_data = [
    ["Node1", "Nuclear"],
    ["Node1", "OCGT"],
    ["Node1", "CCGT"],
    ["Node1", "Solar"],
    ["Node1", "Windonshore"],
    ["Node1", "Windoffshore"],
    ["Node1", "Hydro run-of-the-river"],
]
df_generators_of_node = pd.DataFrame(generators_of_node_data, columns=columns)

df_generators_of_technology = existing_input_client.sets.get_generators_of_technology()
columns = ["Technology", "Generator"]
generators_of_technology_data = [
    ["Gas", "OCGT"],
    ["Gas", "CCGT"],
    ["Solar", "Solar"],
    ["Wind_onshr", "Windonshore"],
    ["Wind_offshr", "Windoffshore"],
    ["Hydro_reg", "Hydro regulated"],
    ["Hydro_ror", "Hydro run-of-the-river"],
    ["Nuclear", "Nuclear"],
]
df_generators_of_technology = pd.DataFrame(generators_of_technology_data, columns=columns)

df_coordinates = existing_input_client.sets.get_coordinates()
columns = ["Location", "Latitude", "Longitude"]
coordinates_data = [
    ["Node1", 58.970156257779884, 5.733379895983701],
    ["NodeDummy", 58.0, 5.0],
]
df_coordinates = pd.DataFrame(coordinates_data, columns=columns)


new_input_client.sets.set_nodes(df_nodes)
new_input_client.sets.set_offshore_nodes(df_offshore_nodes)
new_input_client.sets.set_horizon(df_horizon)
new_input_client.sets.set_storage(df_storage)
new_input_client.sets.set_technology(df_technology)
new_input_client.sets.set_generators(df_generators)
new_input_client.sets.set_line_type(df_line_type)
new_input_client.sets.set_storage_of_nodes(df_storage_of_nodes)
new_input_client.sets.set_directional_lines(df_directional_lines)
new_input_client.sets.set_line_type_of_directional_lines(df_line_type_of_directional_lines)
new_input_client.sets.set_generators_of_node(df_generators_of_node)
new_input_client.sets.set_generators_of_technology(df_generators_of_technology)
new_input_client.sets.set_coordinates(df_coordinates)


################
# TRANSMISSION #
################

df_line_efficiency = existing_input_client.transmission.get_line_efficiency()
columns = ["FromNode", "ToNode", "lineEfficiency"]
line_efficiency_data = [["Node1", "NodeDummy", 0.0], ["NodeDummy", "Node1", 0.0]]
df_line_efficiency = pd.DataFrame(line_efficiency_data, columns=columns)

df_max_built_capacity = existing_input_client.transmission.get_max_built_capacity()
columns = ["InterconnectorLinks", "ToNode", "Period", "TransmissionMaxBuiltCapacity in MW"]
max_built_capacity_data = [["Node1", "NodeDummy", 1, 0.0]]
df_max_built_capacity = pd.DataFrame(max_built_capacity_data, columns=columns)

df_length = existing_input_client.transmission.get_length()
columns = ["FromNode", "ToNode", "lineLength in km"]
length_data = [["Node1", "NodeDummy", 1.0]]
df_length = pd.DataFrame(length_data, columns=columns)

df_type_capital_cost = existing_input_client.transmission.get_type_capital_cost()
columns = ["Type", "Period", "TypeCapitalCost in euro per MWkm"]
type_capital_cost_data = [["HVAC_OverheadLine", 1, 661], ["HVDC_Cable", 1, 2769]]
df_type_capital_cost = pd.DataFrame(type_capital_cost_data, columns=columns)

df_type_fixed_om_cost = existing_input_client.transmission.get_type_fixed_om_cost()
columns = ["Type", "Period", "TypeFixedOMCost in euro per MW"]
type_fixed_om_cost_data = [["HVAC_OverheadLine", 1, 33], ["HVDC_Cable", 1, 138]]
df_type_fixed_om_cost = pd.DataFrame(type_fixed_om_cost_data, columns=columns)

df_initial_capacity = existing_input_client.transmission.get_initial_capacity()
columns = ["InterconnectorLinks", "ToNode", "Period", "TransmissionInitialCapacity"]
initial_capacity_data = [["Node1", "NodeDummy", 1, 0.0]]
df_initial_capacity = pd.DataFrame(initial_capacity_data, columns=columns)

df_max_install_capacity_raw = existing_input_client.transmission.get_max_install_capacity_raw()
columns = ["InterconnectorLinks", "ToNode", "Period", "MaxRawNotAdjustWithInitCap in MW"]
max_install_capacity_raw_data = [["Node1", "NodeDummy", 1, 0.0]]
df_max_install_capacity_raw = pd.DataFrame(max_install_capacity_raw_data, columns=columns)

df_lifetime = existing_input_client.transmission.get_lifetime()
columns = ["InterconnectorLinks", "To Node", "transmissionLifetime"]
lifetime_data = [["Node1", "NodeDummy", 40.0]]
df_lifetime = pd.DataFrame(lifetime_data, columns=columns)


new_input_client.transmission.set_line_efficiency(df_line_efficiency)
new_input_client.transmission.set_max_built_capacity(df_max_built_capacity)
new_input_client.transmission.set_length(df_length)
new_input_client.transmission.set_type_capital_cost(df_type_capital_cost)
new_input_client.transmission.set_type_fixed_om_cost(df_type_fixed_om_cost)
new_input_client.transmission.set_initial_capacity(df_initial_capacity)
new_input_client.transmission.set_max_install_capacity_raw(df_max_install_capacity_raw)
new_input_client.transmission.set_lifetime(df_lifetime)


###########
# STORAGE #
###########

df_initial_power_capacity = existing_input_client.storage.get_initial_power_capacity()
columns = ["Nodes", "StorageTypes", "Period", "InitialPowerCapacity"]
initial_power_capacity_data = [["Node1", "Hydro Pump Storage", 1, 0.0]]
df_initial_power_capacity = pd.DataFrame(initial_power_capacity_data, columns=columns)

df_power_capital_cost = existing_input_client.storage.get_power_capital_cost()
columns = ["StorageTypes", "Period", "PowerCapitalCost in euro per kW"]
power_capital_cost_data = [["Hydro Pump Storage", 1, 2500]]
df_power_capital_cost = pd.DataFrame(power_capital_cost_data, columns=columns)

df_power_fixed_om_cost = existing_input_client.storage.get_power_fixed_om_cost()
columns = ["StorageTypes", "Period", "PowerFixedOMCost in euro per kW"]
power_fixed_om_cost_data = [["Hydro Pump Storage", 1, 0.0]]
df_power_fixed_om_cost = pd.DataFrame(power_fixed_om_cost_data, columns=columns)

df_power_max_built_capacity = existing_input_client.storage.get_power_max_built_capacity()
columns = ["Nodes", "StorageTypes", "Period", "PowerMaxBuiltCapacity"]
power_max_built_capacity_data = [["Node1", "Hydro Pump Storage", 1, 500e3]]
df_power_max_built_capacity = pd.DataFrame(power_max_built_capacity_data, columns=columns)

df_energy_capital_cost = existing_input_client.storage.get_energy_capital_cost()
columns = ["StorageTypes", "Period", "EnergyCapitalCost in euro per kWh"]
energy_capital_cost_data = [["Hydro Pump Storage", 1, 0.0]]
df_energy_capital_cost = pd.DataFrame(energy_capital_cost_data, columns=columns)

df_energy_fixed_om_cost = existing_input_client.storage.get_energy_fixed_om_cost()
columns = ["StorageTypes", "Period", "EnergyFixedOMCost in euro per kWh"]
energy_fixed_om_cost_data = [["Hydro Pump Storage", 1, 0.0]]
df_energy_fixed_om_cost = pd.DataFrame(energy_fixed_om_cost_data, columns=columns)

df_initial_energy_capacity = existing_input_client.storage.get_initial_energy_capacity()
columns = ["Nodes", "StorageTypes", "Period", "EnergyInitialCapacity"]
initial_energy_capacity_data = [["Node1", "Hydro Pump Storage", 1, 0]]
df_initial_energy_capacity = pd.DataFrame(initial_energy_capacity_data, columns=columns)

df_energy_max_built_capacity = existing_input_client.storage.get_energy_max_built_capacity()
columns = ["Nodes", "StorageTypes", "Period", "EnergyMaxBuiltCapacity"]
energy_max_built_capacity_data = [["Node1", "Hydro Pump Storage", 1, 500e3]]
df_energy_max_built_capacity = pd.DataFrame(energy_max_built_capacity_data, columns=columns)

df_energy_max_installed_capacity = existing_input_client.storage.get_energy_max_installed_capacity()
columns = ["Nodes", "StorageTypes", "EnergyMaxInstalledCapacity"]
energy_max_installed_capacity_data = [["Node1", "Hydro Pump Storage", 500e3]]
df_energy_max_installed_capacity = pd.DataFrame(energy_max_installed_capacity_data, columns=columns)

df_power_max_installed_capacity = existing_input_client.storage.get_power_max_installed_capacity()
columns = ["Nodes", "StorageTypes", "PowerMaxInstalledCapacity"]
power_max_installed_capacity_data = [["Node1", "Hydro Pump Storage", 500e3]]
df_power_max_installed_capacity = pd.DataFrame(power_max_installed_capacity_data, columns=columns)

df_storage_initial_energy_level = existing_input_client.storage.get_storage_initial_energy_level()
columns = ["StorageType", "StorageInitialEnergyLevel as a percentage of StorageInstalledEnergyCapacity"]
storage_initial_energy_level_data = [["Hydro Pump Storage", 0.5]]
df_storage_initial_energy_level = pd.DataFrame(storage_initial_energy_level_data, columns=columns)

df_storage_charge_efficiency = existing_input_client.storage.get_storage_charge_efficiency()
columns = ["StorageType", "storageChargeEff"]
storage_charge_efficiency_data = [["Hydro Pump Storage", 0.85]]
df_storage_charge_efficiency = pd.DataFrame(storage_charge_efficiency_data, columns=columns)

df_storage_discharge_efficiency = existing_input_client.storage.get_storage_discharge_efficiency()
columns = ["StorageType", "storageDischargeEff"]
storage_discharge_efficiency_data = [["Hydro Pump Storage", 0.85]]
df_storage_discharge_efficiency = pd.DataFrame(storage_discharge_efficiency_data, columns=columns)

df_storage_power_to_energy = existing_input_client.storage.get_storage_power_to_energy()
columns = ["StorageType", "storagePowToEnergy"]
storage_power_to_energy_data = [["DummyStorage", 1.0]]
df_storage_power_to_energy = pd.DataFrame(storage_power_to_energy_data, columns=columns)

df_storage_bleed_efficiency = existing_input_client.storage.get_storage_bleed_efficiency()
columns = ["StorageType", "storageBleedEff"]
storage_bleed_efficiency_data = [["Hydro Pump Storage", 1.0]]
df_storage_bleed_efficiency = pd.DataFrame(storage_bleed_efficiency_data, columns=columns)

df_lifetime = existing_input_client.storage.get_lifetime()
columns = ["StorageTypes", "storageLifetime"]
lifetime_data = [["Hydro Pump Storage", 30]]
df_lifetime = pd.DataFrame(lifetime_data, columns=columns)


new_input_client.storage.set_initial_power_capacity(df_initial_power_capacity)
new_input_client.storage.set_power_capital_cost(df_power_capital_cost)
new_input_client.storage.set_power_fixed_om_cost(df_power_fixed_om_cost)
new_input_client.storage.set_power_max_built_capacity(df_power_max_built_capacity)
new_input_client.storage.set_energy_capital_cost(df_energy_capital_cost)
new_input_client.storage.set_energy_fixed_om_cost(df_energy_fixed_om_cost)
new_input_client.storage.set_initial_energy_capacity(df_initial_energy_capacity)
new_input_client.storage.set_energy_max_built_capacity(df_energy_max_built_capacity)
new_input_client.storage.set_energy_max_installed_capacity(df_energy_max_installed_capacity)
new_input_client.storage.set_power_max_installed_capacity(df_power_max_installed_capacity)
new_input_client.storage.set_storage_initial_energy_level(df_storage_initial_energy_level)
new_input_client.storage.set_storage_charge_efficiency(df_storage_charge_efficiency)
new_input_client.storage.set_storage_discharge_efficiency(df_storage_discharge_efficiency)
new_input_client.storage.set_storage_power_to_energy(df_storage_power_to_energy)
new_input_client.storage.set_storage_bleed_efficiency(df_storage_bleed_efficiency)
new_input_client.storage.set_lifetime(df_lifetime)


########
# NODE #
########

profiles_path = Path("/Users/martihj/Library/CloudStorage/OneDrive-NTNU/Postdoc/Dataset/DummySystem/loadpvwind8.csv")

df_profiles = pd.read_csv(profiles_path)

df_electric_annual_demand = existing_input_client.nodes.get_electric_annual_demand()
columns = ["Nodes", "Period", "ElectricAdjustment in MWh per hour"]
electric_annual_demand_data = [["Node1", 1, (df_profiles["LOAD"] * 100).sum()], ["NodeDummy", 1, 0.0]]
df_electric_annual_demand = pd.DataFrame(electric_annual_demand_data, columns=columns)

df_node_lost_load_cost = existing_input_client.nodes.get_node_lost_load_cost()
columns = ["Nodes", "Period", "NodeLostLoadCost"]
node_lost_load_cost_data = [["Node1", 1, 3e3], ["NodeDummy", 1, 3e3]]
df_node_lost_load_cost = pd.DataFrame(node_lost_load_cost_data, columns=columns)

df_hydro_generators_max_annual_production = existing_input_client.nodes.get_hydro_generators_max_annual_production()
columns = ["Nodes", "HydroGenMaxAnnualProduction in MWh per year"]
hydro_generators_max_annual_production_data = [["Node1", 0.0]]
df_hydro_generators_max_annual_production = pd.DataFrame(hydro_generators_max_annual_production_data, columns=columns)

new_input_client.nodes.set_electric_annual_demand(df_electric_annual_demand)
new_input_client.nodes.set_node_lost_load_cost(df_node_lost_load_cost)
new_input_client.nodes.set_hydro_generators_max_annual_production(df_hydro_generators_max_annual_production)

#################
# SCENARIO DATA #
#################

datetime_format = "%d/%m/%Y %H:%M"

scenario_data_path = dataset_path / "ScenarioData"
scenario_data_path.mkdir(exist_ok=True, parents=True)

required_files = [
    "windonshore.csv",
    "windoffshore.csv",
    "solar.csv",
    "electricload.csv",
    "hydroror.csv",
    "hydroseasonal.csv",
]
start_time = pd.Timestamp("2020-01-01 00:00")
date_range = pd.date_range(start=start_time, periods=len(df_profiles), freq="1H")

# existing_scenario_data_path = existing_dataset_path / "ScenarioData"

df_windonshore = pd.DataFrame({"time": date_range, "Node1": df_profiles["ONWP"]})
df_windonshore.to_csv(scenario_data_path / "windonshore.csv", index=False, date_format=datetime_format)

df_windoffshore = pd.DataFrame({"time": date_range, "Node1": df_profiles["OFWP"]})
df_windoffshore.to_csv(scenario_data_path / "windoffshore.csv", index=False, date_format=datetime_format)

df_solar = pd.DataFrame({"time": date_range, "Node1": df_profiles["PV"]})
df_solar.to_csv(scenario_data_path / "solar.csv", index=False, date_format=datetime_format)

df_electricload = pd.DataFrame(
    {
        "Node1": df_profiles["LOAD"]*100,
        "time": date_range,
        "hour": date_range.hour,
        "dayofweek": date_range.dayofweek,
        "month": date_range.month,
    }
)

df_electricload.to_csv(scenario_data_path / "electricload.csv", index=False, date_format=datetime_format)

df_hydroror = pd.DataFrame({"time": date_range, "Node1": df_profiles["PV"] * 0.0})  # Dummy
df_hydroror.to_csv(scenario_data_path / "hydroror.csv", index=False, date_format=datetime_format)

df_hydroseasonal = pd.DataFrame({"time": date_range, "Node1": df_profiles["PV"] * 0.0})  # Dummy
df_hydroseasonal.to_csv(scenario_data_path / "hydroseasonal.csv", index=False, date_format=datetime_format)

summer_start = pd.Timestamp("2020-06-01 00:00")
summer_end = pd.Timestamp("2020-10-01 00:00")

summer_range = pd.date_range(summer_start, summer_end, freq="1H")
n_summer_steps = len(summer_range)
