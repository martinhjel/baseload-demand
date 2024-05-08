# Snakefile

import itertools
import pandas as pd
from pathlib import Path

EMPIRE_PATH = Path("../OpenEMPIRE")
RESULTS_PATH = EMPIRE_PATH / "Results/genesis_ramp/1_node_baseload"
RESULTS_WIND_PATH = RESULTS_PATH / "wind"

# Define parameters
parameters = {
    "nuclear_capital_cost": [5000, 6000,7000], # int
    "co2_price": [150], # int
    "load_scaling": [1.], #  float 
    "load_shifting": [-40 + i for i in range(0, 150, 10)], # int
    "ramp_rate": [0.85]
}

# Generate combinations of parameters
cases = list(itertools.product(*parameters.values()))

# Rule for running the script
rule all:
    input:
        expand(RESULTS_PATH/"ncc_{ncc}_co2_{co2}_scale_{scale}_shift{shift}_ramp{ramp}/.done", 
               ncc=[p[0] for p in cases], 
               co2=[p[1] for p in cases], 
               scale=[p[2] for p in cases], 
               shift=[p[3] for p in cases],
               ramp=[p[4] for p in cases])
rule run_model:
    output:
        RESULTS_PATH/"ncc_{ncc}_co2_{co2}_scale_{scale}_shift{shift}_ramp{ramp}/.done"
    params:
        ncc="{ncc}",
        co2="{co2}",
        scale="{scale}",
        shift="{shift}",
        ramp="{ramp}"
    log:
        "logs/ncc_{ncc}_co2_{co2}_scale_{scale}_shift{shift}.log"
    shell:
        "python scripts/baseload_analysis.py -ncc {params.ncc} -co2 {params.co2} -scale {params.scale} -shift {params.shift} -r {params.ramp}"

rule all_wind_scenarios:
    input:
        [RESULTS_WIND_PATH/f"ncc_6000_co2_150_scale_1.0_shift0_ramp0.85_wind{wind}/.done" for wind in range(1986,2016)]

rule dag:
    message:
        "Creating DAG of workflow."
    output:
        dot="dag.dot",
        pdf="dag.pdf",
        png="dag.png",
    shell:
        """
        snakemake --dag all | dot -Tpdf -o {output.pdf} {output.dot}
        dot -Tpng -o {output.png} {output.dot}
        """
