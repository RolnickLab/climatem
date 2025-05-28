# NOTE: as of 14th Oct, I am also trying to get this to work for multiple variables.

import glob
import os
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union


import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="cfgrib")

# from climatem.constants import ( 
#     ERA5_NOM_RES, # CHRISTINA QUESTION: SEE constants.py
#     ERA5_TEMP_RES, # CHRISTINA QUESTION: this is "day"?
# )

# from climatem.plotting.plot_data import plot_species, plot_species_anomaly
from climatem.utils import get_logger
from .climate_dataset import ClimateDataset

log = get_logger()

# base data set: implements copy to slurm, get item etc pp
# era5 data set: model wise
# from datamodule create one of these per train/test/val


class ERA5Dataset(ClimateDataset):
    """
    Load in reanalysis data from ERA5 for seasonal forecast.
    To decide:
    - taking in daily data, daily predictions?
    - No "scenarios" so num_scenarios=1?
    Load scenarios for train/val: Take this as a list Process and save this as .npz in $SLURM_TMPDIR Load these in
    train/val/test Dataloader functions.

    Keep one scenario for testing # Target shape (no. yrs * no. mon, no. vars, lat, lon) # ! * num_scenarios!!
    """

    def __init__(  # noqa: C901
        # inherits all the stuff from Base
        self,
        years: Union[int, str],
        historical_years: Union[int, str],
        data_dir: Optional[str] = "data",
        climate_model: str = "ERA5",
        num_ensembles: int = 0,  # 1 for first ensemble, -1 for all
        scenarios: List[str] = [""],
        variables: List[str] = ["t2m"],
        mode: str = "train",
        output_save_dir: str = "/home/mila/l/lastc/scratch/results/diagnostics/",
        reload_climate_set_data: bool = True,
        channels_last: bool = True,
        seq_to_seq: bool = True,
        global_normalization: bool = True,
        seasonality_removal: bool = True,
        seq_len: int = 365,
        lat: int = 96, 
        lon: int = 144,
        icosahedral_coordinates_path: str = "/mappings/vertex_lonlat_mapping.txt", # CHRISTINA QUESTION: current location, can change to mappings, does it need to by npy?
        *args,
        **kwargs,
    ):  # noqa: C901

        self.mode = mode
        self.output_save_dir = Path(output_save_dir)
        self.reload_climate_set_data = reload_climate_set_data
        self.root_dir = Path(data_dir)
        self.input_nc_files = []
        self.output_nc_files = []
        self.in_variables = variables
        self.global_normalization = global_normalization
        self.seasonality_removal = seasonality_removal
        self.seq_len = seq_len
        self.lon = lon
        self.lat = lat
        self.icosahedral_coordinates_path = icosahedral_coordinates_path

        fname_kwargs = dict(
            climate_model=climate_model,
            num_ensembles=num_ensembles,
            years=f"{years[0]}-{years[-1]}",
            historical_years=f"{historical_years[0]}-{historical_years[-1]}",
            variables=variables,
            scenarios=scenarios,
            channels_last=channels_last,
            seq_to_seq=seq_to_seq,
        )
        self.fname_kwargs = fname_kwargs

        print("self.fname_kwargs instantiated")

        # TO-DO: This is just getting the list of .nc files for targets. Put this logic in a function and get input list as well.
        # In a function, we can call CausalDataset() instance for train and test separately to load the data

        if isinstance(climate_model, str):
            self.root_dir = self.root_dir / climate_model
        else:
            # Logic for multiple climate models, not sure how to load/create dataset yet
            log.warn("Data loader not yet implemented for multiple climate models.")
            raise NotImplementedError

        # I am actually going to make this a list to be compatible with the rest of the code
        if num_ensembles == 1:
            ensemble = os.listdir(self.root_dir)
            # if there is only one element in the ensemble list, we can just take the first element
            if len(ensemble) == 1:
                # print("ensemble:", ensemble)
                # print("This often makes a mistake because it does not know if it wants to be a list or not")
                self.ensemble_dirs = [self.root_dir / ensemble[0]]
            else:  # we are just going to select the first ensemble member here
                self.ensemble_dirs = [
                    self.root_dir / ensemble[0]
                ]  # THIS USED TO BE THE CASE: Taking specific ensemble member (#TODO: only this ensemble member has historical data...)
        elif num_ensembles == 0:
            self.ensemble_dirs = [self.root_dir]
        else:
            # TODO elif self.reload_climate_set_data
            log.warn(
                "Data loader not properly yet implemented for multiple ensemble members, but we are trying something here."
            )
            # here I want to make the dataloader work for all ensemble members:
            # I need to loop through all the ensemble members and load the data
            ensembles = os.listdir(self.root_dir)
            # print("Ensemble members present for this model:", ensembles)
            # Now make a list, which consists of the paths to the ensemble members
            self.ensemble_dirs = [self.root_dir / ensemble for ensemble in ensembles]

            # print("Ensemble directories:", self.ensemble_dirs)
            # print("What is the type of self.ensemble_dirs:", type(self.ensemble_dirs))

        fname, coordinates_fname = self.get_save_name_from_kwargs(mode=mode, file="target", kwargs=self.fname_kwargs)
        print(f"coordinates_fname {coordinates_fname}")

        # here we reload files if they exist
        if (
            os.path.isfile(self.output_save_dir / fname) and self.reload_climate_set_data
        ):  # we first need to get the name here to test that...

            self.data_path = self.output_save_dir / fname
            # print("path exists, reloading")
            self.raw_data = self._reload_data(self.data_path)
            self.coordinates = self.load_dataset_coordinates(coordinates_fname, mode=self.mode, mips="cmip6")

            if self.global_normalization:
                # Load stats and normalize
                stats_fname, coordinates_fname = self.get_save_name_from_kwargs(
                    mode=mode, file="statistics", kwargs=self.fname_kwargs
                )
                stats = self.load_dataset_statistics(stats_fname, mode=self.mode, mips="cmip6")
                self.Data = self.normalize_data(self.raw_data, stats)
            else:
                self.Data = self.raw_data
            if self.seasonality_removal:
                self.Data = self.remove_seasonality(self.Data)

            # print("In CMIP6Dataset, just finished removing the seasonality.")

        else:
            print("NOT RELOADING!!!")
            # Add code here for adding files for input nc data
            # Similar to the loop below for output files

            # Got all the files paths at this point, now open and merge

            # List of output files
            files_per_var = []
            for var in variables:

                for exp in scenarios:
                    if exp == "historical":
                        get_years = historical_years
                    else:
                        get_years = years
                    # print("ensemble_dirs")
                    # print(self.ensemble_dirs)

                    all_ensemble_output_nc_files = []

                    # print("What is the type of self.ensemble_dirs:", type(self.ensemble_dirs))

                    # assert that self.ensemble_dirs is a list
                    if isinstance(self.ensemble_dirs, list):
                        print("self.ensemble_dirs is a list")
                    else:
                        # print("self.ensemble_dirs is not a list")
                        # print("self.ensemble_dirs is:", self.ensemble_dirs)
                        raise ValueError("self.ensemble_dirs is not a list")

                    for ensemble_dir in self.ensemble_dirs:
                        print("*****************LOOPING THROUGH ENSEMBLE MEMBERS*****************")

                        # I am now identing this:
                        output_nc_files = []

                        for y in get_years:
                            # for y in self.get_years_list(get_years, give_list=True):
                            # print('y is this:', y)
                            # print('here is exp:', exp)
                            ensemble_dir = Path(ensemble_dir)
                            var_dir = ensemble_dir / f"{exp}/ERA5_All_Variables_{y}/{y}/daily/gme24"
                            print(f"ALL FILES DIRECTORY: {var_dir}")
                            files = glob.glob(f"{var_dir}/*.nc", recursive=True)
                            print(f"NC FILES: {files}")
                            if len(files) == 0:
                                # print(f"No netcdf files found in {var_dir}, trying to find .grib files")
                                files = glob.glob(f"{var_dir}/*.grib2", recursive=True)
                            print(f"Grib FILES: {files}")
                            # print('files here:', files)
                            # loads all years! implement splitting
                            output_nc_files += files

                        # print("Here the final var_dir be:", var_dir)
                        # print('files here after looping through all the years:', output_nc_files)
                        # print(
                        #     "length of output_nc_files. after looping through years for 1 of the ensemble members:",
                        #     len(output_nc_files),
                        # )

                        all_ensemble_output_nc_files += output_nc_files

                    # print("Here the final var_dir be:", var_dir)
                    # print(
                    #     "length of all_ensemble_output_nc_files after looping through all ensemble members:",
                    #     len(all_ensemble_output_nc_files),
                    # )
                    # print('files here after looping through all the ensembles and the years:', all_ensemble_output_nc_files)
                files_per_var.append(all_ensemble_output_nc_files)
            print("length of files_per_var after looping!:", len(files_per_var))
            print('files_per_var:', files_per_var)

            # self.raw_data_input = self.load_data_into_mem(self.input_nc_files) #currently don't have input paths etc
            self.raw_data = self.load_into_mem(
                files_per_var, num_vars=len(variables), channels_last=channels_last, seq_to_seq=seq_to_seq,
            )
            self.coordinates = self.load_coordinates_into_mem(files_per_var)

            if self.mode == "train" or self.mode == "train+val":
                print("creating stats fname")
                print(f"self.fname_kwargs {self.fname_kwargs}")
                stats_fname, coordinates_fname = self.get_save_name_from_kwargs(
                    mode=mode, file="statistics", kwargs=self.fname_kwargs
                )
                print("creating stats / coordinates name")
                print(stats_fname)
                print(coordinates_fname)

                if os.path.isfile(stats_fname) and self.global_normalization:
                    print("Stats file already exists! Loading from memory.")
                    stats = self.load_dataset_statistics(stats_fname, mode=self.mode, mips="cmip6")
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
                    )
                    stats = self.load_dataset_statistics(stats_fname, mode=self.mode, mips="cmip6")
                    self.norm_data = self.normalize_data(self.raw_data, stats)
                else:
                    self.norm_data = self.raw_data
                if self.seasonality_removal:
                    self.norm_data = self.remove_seasonality(self.norm_data)
            # self.input_path = self.save_data_into_disk(self.raw_data_input, self.mode, 'input')
            # print("In cmip6, just about to save the data.")
            self.data_path = self.save_data_into_disk(self.raw_data, fname, self.output_save_dir)
            # print("In cmip6, just saved the data.")

            # print("In cmip6, just about to copy the data to slurm.")
            # self.copy_to_slurm(self.input_path)
            self.copy_to_slurm(self.data_path)
            # print("In cmip6, just copied the data to slurm.")

            self.Data = self.norm_data

        # plot_species(self.Data[:, :, 0, :, :], self.coordinates, variables, "../../TEST_REPO", "before_causal")
        # self.Data = self._reload_data(self.data_path)

        # Now X and Y is ready for getitem
        # print("CMIP6 shape", self.Data.shape)
        self.length = self.Data.shape[0]

    def __getitem__(self, index):
        return self.Data[index]