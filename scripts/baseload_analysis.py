import argparse
from pathlib import Path

from empire.core.config import EmpireConfiguration, read_config_file
from empire.core.model_runner import run_empire_model, setup_run_paths
from empire.input_client.client import EmpireInputClient
from empire.input_data_manager import (
    CapitalCostManager,
    CO2PricetManager,
    ElectricLoadManager,
    RampRateManager,
)
from empire.logger import get_empire_logger

## Read config and setup folders ##
config = read_config_file(Path("config/run_1_node.yaml"))
empire_config = EmpireConfiguration.from_dict(config=config)

parser = argparse.ArgumentParser(description="Run baseload analysis")
parser.add_argument(
    "-ncc",
    "--nuclear_capital_cost",
    type=float,
    required=True,
    help="Nuclear capital cost",
)
parser.add_argument("-co2", "--co2-price", type=float, required=True, help="CO2 price")
parser.add_argument("-scale", "--load-scaling", type=float, required=True, help="Load scaling factor")
parser.add_argument("-shift", "--load-shifting", type=float, required=True, help="Load shifting value")
parser.add_argument("-ramp", "--ramp-rate", type=float, required=True, help="Ramp rate value of nuclear")
parser.add_argument("-t", "--test-run", action="store_true", help="Dont run optimization")

# Parse arguments
args = parser.parse_args()
ncc = int(args.nuclear_capital_cost)
co2 = int(args.co2_price)
scale = float(args.load_scaling)
shift = int(args.load_shifting)
ramp_rate = float(args.ramp_rate)

run_path = Path("../OpenEMPIRE") / "Results/genesis/1_node_baseload/"
run_path = run_path / f"ncc_{ncc}_co2_{co2}_scale_{scale}_shift{shift}_ramp{ramp_rate}"

run_config = setup_run_paths(
    version="1_node",
    empire_config=empire_config,
    run_path=run_path,
    empire_path=Path("../OpenEMPIRE"),
)
logger = get_empire_logger(run_config=run_config)

logger.info("Running EMPIRE Model")

client = EmpireInputClient(dataset_path=run_config.dataset_path)
data_managers = [
    CapitalCostManager(client=client, generator_technology="Nuclear", capital_cost=ncc),
    CO2PricetManager(client=client, periods=[1], co2_prices=[co2]),
    ElectricLoadManager(client=client, run_config=run_config, scale=scale, shift=shift, node="Node1"),
    RampRateManager(client=client, thermal_generator="Nuclear", ramp_rate=ramp_rate)
]

## Run empire model
run_empire_model(
    empire_config=empire_config,
    run_config=run_config,
    data_managers=data_managers,
    test_run=args.test_run,
)

# Create a marker file to indicate completion
with open(run_path / ".done", "w") as f:
    f.write("done")
