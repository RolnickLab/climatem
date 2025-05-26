# NOTE: as of 14th Oct, I am also trying to get this to work for multiple variables.

import glob
import os
import zipfile
from pathlib import Path
from typing import List, Optional, Tuple, Union

from climatem.constants import (  # INPUT4MIPS_NOM_RES,; INPUT4MIPS_TEMP_RES,
    CMIP6_NOM_RES,
    CMIP6_TEMP_RES,
    NO_OPENBURNING_VARS,
)

# from climatem.plotting.plot_data import plot_species, plot_species_anomaly
from climatem.utils import get_logger

from .climate_dataset import ClimateDataset

log = get_logger()


# base data set: implements copy to slurm, get item etc pp
# input4mips data set: same per model
# from datamodule create one of these per train/test/val

class Input4MipsDataset(ClimateDataset):
    """
    Loads all scenarios for a given var / for all vars.

    TODO: Are coordinates correct here? Rather than reloading, should use the vertez mapping .npy file
    """

    def __init__(  # inherits all the stuff from Base
        self,
        years: Union[int, str],
        historical_years: Union[int, str],
        data_dir: Optional[str] = "Climateset_DATA",
        variables: List[str] = ["BC_sum"],
        scenarios: List[str] = ["ssp126", "ssp370", "ssp585"],
        channels_last: bool = False,
        openburning_specs: Tuple[str] = ("no_fires", "no_fires"),
        mode: str = "train",
        output_save_dir: str = "",
        reload_climate_set_data: bool = True,
        global_normalization: bool = True,
        seasonality_removal: bool = True,
        seq_len: int = 12,
        lat: int = 96,
        lon: int = 144,
        icosahedral_coordinates_path: str = "/mappings/vertex_lonlat_mapping.npy",
        *args,
        **kwargs,
    ):

        self.channels_last = channels_last

        self.mode = mode
        self.root_dir = Path(data_dir) / "inputs/input4mips"
        self.output_save_dir = Path(output_save_dir)
        self.reload_climate_set_data = reload_climate_set_data
        self.input_nc_files = []
        self.output_nc_files = []
        self.global_normalization = global_normalization
        self.seasonality_removal = seasonality_removal
        self.variables = variables
        self.seq_len = seq_len
        self.lon = lon
        self.lat = lat
        self.icosahedral_coordinates_path = icosahedral_coordinates_path

        if len(historical_years) == 0:
            historical_years_str = "no_historical"
        elif len(historical_years) == 1:
            historical_years_str = f"{historical_years[0]}"
        else:
            historical_years_str = f"{historical_years[0]}-{historical_years[-1]}"

        fname_kwargs = dict(
            years=f"{years[0]}-{years[-1]}",
            historical_years=historical_years_str,
            variables=variables,
            scenarios=scenarios,
            channels_last=channels_last,
            openburning_specs=openburning_specs,
            seq_to_seq=True,
        )
        self.fname_kwargs = fname_kwargs

        historical_openburning, ssp_openburning = openburning_specs

        # Split the data here using n_years if needed,
        # else do random split logic here
        fname, coordinates_fname = self.get_save_name_from_kwargs(mode=mode, file="input", kwargs=fname_kwargs)

        # Check here if os.path.isfile(data.npz) exists #TODO: check if exists on slurm
        # if it does, use self._reload data(path)

        if (
            os.path.isfile(output_save_dir / fname) and self.reload_climate_set_data
        ):  # we first need to get the name here to test that...
            self.data_path = output_save_dir / fname
            # print("path exists, reloading")
            self.raw_data = self._reload_data(self.data_path)
            self.coordinates = self.load_dataset_coordinates(coordinates_fname, mode=self.mode, mips="input4mips")

            # Load stats and normalize
            if self.global_normalization:
                stats_fname, coordinates_fname = self.get_save_name_from_kwargs(
                    mode=mode, file="statistics", kwargs=fname_kwargs
                )
                stats = self.load_dataset_statistics(stats_fname, mode=self.mode, mips="input4mips")
                self.Data = self.normalize_data(self.raw_data, stats)
            else:
                self.Data = self.raw_data
            if self.seasonality_removal:
                self.Data = self.remove_seasonality(self.Data)

            # print("In Input4mips, just finished removing the seasonality.")

        else:
            files_per_var = []
            for var in variables:
                # print("var", var)
                output_nc_files = []

                for exp in scenarios:  # TODO: implement getting by years! also sub selection for historical years
                    # print("exp", exp)
                    if var in NO_OPENBURNING_VARS and exp == "historical":
                        # print("I am in var in no_openburningvars and historical in input4mips")
                        filter_path_by = ""
                        get_years = historical_years
                    elif var in NO_OPENBURNING_VARS:
                        filter_path_by = ""
                        get_years = years
                    elif exp == "historical":
                        # print("I am in historical in input4mips")
                        filter_path_by = historical_openburning
                        get_years = historical_years
                    else:
                        # print("I am in else in INPUT4MIPS")
                        filter_path_by = ssp_openburning
                        get_years = years

                    for y in get_years:
                        # print('Input4mips y:', y )
                        var_dir = self.root_dir / f"{exp}/{var}/{CMIP6_NOM_RES}/{CMIP6_TEMP_RES}/{y}"
                        files = list(var_dir.rglob(f"*{filter_path_by}*.nc"))
                        # print('files in input4mips', files)
                        output_nc_files += files
                files_per_var.append(output_nc_files)

            self.raw_data = self.load_into_mem(
                files_per_var, num_vars=len(variables), channels_last=self.channels_last, seq_to_seq=True
            )  # we always want the full sequence for input4mips
            self.coordinates = self.load_coordinates_into_mem(files_per_var)

            if self.mode == "train" or self.mode == "train+val":
                stats_fname, coordinates_fname = self.get_save_name_from_kwargs(
                    mode=mode, file="statistics", kwargs=fname_kwargs
                )

                if os.path.isfile(stats_fname) and self.global_normalization:
                    print("Stats file already exists! Loading from memory.")
                    stats = self.load_dataset_statistics(stats_fname, mode=self.mode, mips="input4mips")
                    self.norm_data = self.normalize_data(self.raw_data, stats)
                elif self.global_normalization:
                    stat1, stat2 = self.get_dataset_statistics(self.raw_data, self.mode, mips="cmip6")
                    stats = {"mean": stat1, "std": stat2}
                    self.norm_data = self.normalize_data(self.raw_data, stats)
                    self.write_dataset_statistics(stats_fname, stats)
                    self.write_dataset_statistics(coordinates_fname, self.coordinates)
                else:
                    self.norm_data = self.raw_data
                if self.seasonality_removal:
                    self.norm_data = self.remove_seasonality(self.norm_data)

                # self.norm_data = self.normalize_data(self.raw_data, stats)

            elif self.mode == "test":
                if self.global_normalization:
                    stats_fname, coordinates_fname = self.get_save_name_from_kwargs(
                        mode="train+val", file="statistics", kwargs=fname_kwargs
                    )  # Load train stats cause we don't calculcate norm stats for test.
                    stats = self.load_dataset_statistics(stats_fname, mode=self.mode, mips="input4mips")
                    self.norm_data = self.normalize_data(self.raw_data, stats)
                else:
                    self.norm_data = self.raw_data
                if self.seasonality_removal:
                    self.norm_data = self.remove_seasonality(self.norm_data)

            # self.input_path = self.save_data_into_disk(self.raw_data_input, self.mode, 'input')
            # print("In input4mips, just about to save the data.")
            self.data_path = self.save_data_into_disk(self.raw_data, fname, self.output_save_dir)
            # print("In input4mips, just saved the data.")

            # print("In input4mips, just about to copy the data to slurm.")
            # self.copy_to_slurm(self.input_path)
            self.copy_to_slurm(self.data_path)
            # print("In input4mips, just copied the data to slurm.")

            # Call _reload_data here with self.input_path and self.output_path
            # self.X = self._reload_data(input_path)
            self.Data = self.norm_data
            # self.Data = self._reload_data(self.data_path)
            # Write a normalize transform to calculate mean and std
            # Either normalized whole array here or per instance getitem, that maybe faster

            # Now X and Y is ready for getitem
        # print("Input4mips shape", self.Data.shape)
        self.length = self.Data.shape[0]

    def __getitem__(self, index):
        return self.Data[index]

    def __len__(self):
        return self.length
