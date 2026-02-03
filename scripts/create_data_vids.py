import glob
import os
from pathlib import Path
from typing import List, Optional, Union
import itertools
import zipfile
from typing import Dict, List, Optional, Union  # Tuple

import numpy as np
import torch
import xarray as xr

from climatem.data_loader.healpix_remapping import remap_reg_to_healpix
from climatem.utils import get_logger
from climatem.data_loader.climate_dataset import ClimateDataset

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

from climatem.constants import AVAILABLE_MODELS_FIRETYPE, OPENBURNING_MODEL_MAPPING
from climatem.constants import (  # INPUT4MIPS_NOM_RES,; INPUT4MIPS_TEMP_RES,
    CMIP6_NOM_RES,
    CMIP6_TEMP_RES,
    NO_OPENBURNING_VARS,
)

from climatem.data_loader.input4mip_dataset import Input4MipsDataset
from climatem.plotting.plot_data import plot_species

if __name__ == "__main__":

    input4mips_data = Input4MipsDataset(
        years = "2015-2100",
        historical_years = "1850-2015",
        data_dir = "/home/mila/j/julien.boussard/scratch/Climateset_DATA/huggingface_data/",
        variables = ["BC_sum", "SO2_sum", "CH4_sum", "CO2_sum"],
        scenarios = ["historical", "ssp126", "ssp245"],
        output_save_dir = "/home/mila/j/julien.boussard/scratch/Climateset_DATA/input4mips_numpy",
        reload_climate_set_data = False,
        global_normalization = False,
        seasonality_removal = False,
        map_to_healpix = False,
    )

    out_dir = "/home/mila/j/julien.boussard/scratch/Climateset_DATA/input4mips_numpy"
    list_var = ["BC_sum", "SO2_sum", "CH4_sum", "CO2_sum"]

    name_video = "historical_data"
    plot_species(input4mips_data.raw_data[:165], 
                input4mips_data.coordinates,
                list_var, 
                out_dir, 
                name_video,
                year_range = 165,
                start_year = 1850,
                cmap = 'Reds'
                )

    name_video = "ssp126"
    plot_species(input4mips_data.raw_data[165:251], 
                input4mips_data.coordinates,
                list_var, 
                out_dir, 
                name_video,
                year_range = 86,
                start_year = 2015,
                cmap = 'Reds'
                )

    name_video = "ssp245"
    plot_species(input4mips_data.raw_data[251:], 
                input4mips_data.coordinates,
                list_var, 
                out_dir, 
                name_video,
                year_range = 86,
                start_year = 2015,
                cmap = 'Reds'
                )



