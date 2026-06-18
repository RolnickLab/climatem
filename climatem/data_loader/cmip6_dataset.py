# NOTE: as of 14th Oct, I am also trying to get this to work for multiple variables.

import glob
import os
from pathlib import Path
from typing import List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from climatem.constants import (
    AERMON_VARIABLES,
    GLOBAL_FORCING_VARIABLES,
    GM_VARIABLES,
    MODEL_DB_MAPPING,
    SPATIAL_FORCING_VARIABLES,
    SPATIAL_VARIABLES,
    TARGET_VARIABLES,
)
from climatem.data_loader.climate_dataset import ClimateDataset

# from climatem.constants import (  # INPUT4MIPS_NOM_RES,; INPUT4MIPS_TEMP_RES,
#     CMIP6_NOM_RES,
#     CMIP6_TEMP_RES,
# ) # ADD constants here for later??
from climatem.data_loader.healpix_remapping import remap_reg_to_healpix

# from climatem.plotting.plot_data import plot_species, plot_species_anomaly
from climatem.utils import get_logger

log = get_logger()

# base data set: implements copy to slurm, get item etc pp
# cmip6 data set: model wise
# from datamodule create one of these per train/test/val


class CMIP6Dataset(ClimateDataset):
    """
    Use first ensemble member for now Option to use multile ensemble member later Give option for which variable to use
    Load 3 scenarios for train/val: Take this as a list Process and save this as .npz in $SLURM_TMPDIR Load these in
    train/val/test Dataloader functions.

    Keep one scenario for testing # Target shape (85 * 12, 1, 144, 96) # ! * num_scenarios!!
    """

    def __init__(  # noqa: C901
        # inherits all the stuff from Base
        self,
        years: Union[int, str],
        historical_years: Union[int, str],
        data_dir: Optional[str] = "Climateset_DATA",
        climate_model: str = "NorESM2-LM",
        num_ensembles: int = 1,  # 1 for first ensemble, -1 for all
        scenarios: List[str] = ["ssp245"],  # Right now only one scenario is supported
        variables: List[str] = ["pr"],
        mode: str = "train",
        output_save_dir: str = "",
        reload_climate_set_data: bool = True,
        channels_last: bool = True,
        seq_to_seq: bool = True,
        map_to_healpix: bool = True,
        global_normalization: bool = True,
        seasonality_removal: bool = True,
        temp_res: str = "mon",
        seq_len: int = 12,
        lat: int = 96,
        lon: int = 144,
        icosahedral_coordinates_path: str = "../../mappings/vertex_lonlat_mapping.npy",
        *args,
        **kwargs,
    ):  # noqa: C901

        self.mode = mode
        self.output_save_dir = Path(output_save_dir)
        self.reload_climate_set_data = reload_climate_set_data
        self.root_dir = Path(data_dir)  # / "outputs/CMIP6" (Previously for already preprocessed data)
        self.input_nc_files = []
        self.output_nc_files = []
        self.in_variables = variables
        self.map_to_healpix = map_to_healpix
        self.global_normalization = global_normalization
        self.seasonality_removal = seasonality_removal
        self.temp_res = temp_res
        self.seq_len = seq_len
        self.lon = lon
        self.lat = lat
        self.icosahedral_coordinates_path = icosahedral_coordinates_path
        print(f"CMIP6 self.icosahedral_coordinates_path {self.icosahedral_coordinates_path}")

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

        # print("IN CMIP6!!!")

        if isinstance(climate_model, str) and climate_model == "NorESM2-LM":
            database = MODEL_DB_MAPPING[climate_model]
        elif isinstance(climate_model, str):
            log.warn("Data loader only implemented for NorESM2-LM yet.")
            raise NotImplementedError
        else:
            # Logic for multiple climate models, not sure how to load/create dataset yet
            log.warn("Data loader not yet implemented for multiple climate models.")
            raise NotImplementedError

        if len(scenarios) == 1:
            if scenarios[0] in ["historical", "piControl"]:
                id_scen = "CMIP"
            else:
                id_scen = "ScenarioMIP"
        else:
            # Logic for multiple scenarios, not sure how to load/create dataset yet
            log.warn("Data loader not yet implemented for multiple scenarios.")
            raise NotImplementedError

        self.root_dir = self.root_dir / f"{id_scen}/{database}/{climate_model}/{scenarios[0]}"

        # I am actually going to make this a list to be compatible with the rest of the code
        if num_ensembles == 1:
            ensemble = os.listdir(self.root_dir)
            self.ensemble_dirs = self.root_dir / ensemble[0]
            # if there is only one element in the ensemble list, we can just take the first element
            # if len(ensemble) == 1:
            #     self.ensemble_dirs = self.root_dir / ensemble[0]
            # else:  # we are just going to select the first ensemble member here
            #     self.ensemble_dirs = [
            #         self.root_dir / ensemble[0]
            #     ]  # THIS USED TO BE THE CASE: Taking specific ensemble member (#TODO: only this ensemble member has historical data...)
        else:
            # TODO elif self.reload_climate_set_data
            log.warn(
                "Data loader not properly yet implemented for multiple ensemble members, but we are trying something here."
            )
            raise NotImplementedError
            # ensembles = os.listdir(self.root_dir)
            # self.ensemble_dirs = [self.root_dir / ensemble for ensemble in ensembles]

            # print("Ensemble directories:", self.ensemble_dirs)
            # print("What is the type of self.ensemble_dirs:", type(self.ensemble_dirs))

        if self.temp_res == "mon":
            res_map = []
            spatial_res = []
            for var in self.in_variables:
                res_map.append("AERmon" if var in AERMON_VARIABLES else "Amon")
                spatial_res.append("gm" if var in GM_VARIABLES else "gn")

            self.ensemble_dirs = [
                self.ensemble_dirs / f"{res}/{var}/{spatialres}"
                for var, res, spatialres in zip(self.in_variables, res_map, spatial_res)
            ]
        else:
            # Logic for multiple scenarios, not sure how to load/create dataset yet
            log.warn("Data loader not yet implemented for multiple scenarios.")
            raise NotImplementedError

        # TODO: Need to implement version + grid for better path control BOTTOM IS HARDCODED
        # self.ensemble_dirs = [ensemble_dir / "v20210118" for ensemble_dir in self.ensemble_dirs] #v20190815

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
            self.Data = self.Data.astype("float32")
            # print("In CMIP6Dataset, just finished removing the seasonality.")

        else:
            print("NOT RELOADING!!!")
            # Add code here for adding files for input nc data
            # Similar to the loop below for output files

            # Got all the files paths at this point, now open and merge

            # List of output files
            files_per_var = []
            # for var in variables:

            #     for exp in scenarios:
            # if exp == "historical":
            #     get_years = historical_years
            # else:
            #     get_years = years
            # print("ensemble_dirs")
            # print(self.ensemble_dirs)

            # all_ensemble_output_nc_files = []

            # print("What is the type of self.ensemble_dirs:", type(self.ensemble_dirs))

            # assert that self.ensemble_dirs is a list
            if isinstance(self.ensemble_dirs, list):
                print("self.ensemble_dirs is a list")
            else:
                # print("self.ensemble_dirs is not a list")
                # print("self.ensemble_dirs is:", self.ensemble_dirs)
                raise ValueError("self.ensemble_dirs is not a list")

            print("*****************LOOPING THROUGH VARIABLES *****************")
            for ensemble_dir, var in zip(self.ensemble_dirs, self.in_variables):

                print("ensemble member path:", ensemble_dir)

                # I am now identing this:
                # output_nc_files = []

                # for y in get_years:
                # for y in self.get_years_list(get_years, give_list=True):
                # print('y is this:', y)
                # print('here is exp:', exp)
                # var_dir = ensemble_dir # TODO: This should be rewritten according to ESMValTools / f"{exp}/{var}/{CMIP6_NOM_RES}/{CMIP6_TEMP_RES}/{y}"
                # print(f"ALL FILES DIRECTORY: {ensemble_dir}")
                files = glob.glob(f"{ensemble_dir}/**/*.nc", recursive=True)
                if len(files) == 0:
                    # print(f"No netcdf files found in {var_dir}, trying to find .grib files")
                    files = glob.glob(f"{ensemble_dir}/*.grib", recursive=True)
                files = sorted(files)
                # print('files here:', files)
                # loads all years! implement splitting
                # output_nc_files += files

                # print("Here the final var_dir be:", var_dir)
                # print('files here after looping through all the years:', output_nc_files)
                # print(
                #     "length of output_nc_files. after looping through years for 1 of the ensemble members:",
                #     len(output_nc_files),
                # )

                # all_ensemble_output_nc_files.append(files)

                # print("Here the final var_dir be:", var_dir)
                # print(
                #     "length of all_ensemble_output_nc_files after looping through all ensemble members:",
                #     len(all_ensemble_output_nc_files),
                # )
                # print('files here after looping through all the ensembles and the years:', all_ensemble_output_nc_files)
                files_per_var.append(files)
            # files_per_var.append(all_ensemble_output_nc_files)
            # print("length of files_per_var after looping!:", len(files_per_var))
            # print('files_per_var:', files_per_var)

            # self.raw_data_input = self.load_data_into_mem(self.input_nc_files) #currently don't have input paths etc
            self.raw_data = self.load_into_mem(
                files_per_var,
                num_vars=len(variables),
                channels_last=channels_last,
                seq_to_seq=seq_to_seq,
                variables=self.in_variables,
            )  # (86, 12, 1, 96, 144)

            lon, lat = self.load_coordinates_into_mem(files_per_var)

            if self.map_to_healpix:
                print("remapping to healpix")
                self.raw_data, latitudes_new, longitudes_new = remap_reg_to_healpix(
                    self.raw_data, lon, lat
                )  # (86, 12, 1, 3072)
                self.coordinates = np.c_[longitudes_new, latitudes_new]
            else:
                self.coordinates = self.load_coordinates_into_mem(files_per_var)
                self.coordinates = np.meshgrid(self.coordinates[0], self.coordinates[1])
                self.coordinates = np.c_[self.coordinates[1].flatten(), self.coordinates[0].flatten()]

            if self.mode in ["train", "train+val"]:
                print("creating stats fname")
                print(f"self.fname_kwargs {self.fname_kwargs}")
                stats_fname, coordinates_fname = self.get_save_name_from_kwargs(
                    mode=mode, file="statistics", kwargs=self.fname_kwargs
                )
                print("creating stats / coordinates name")
                print(stats_fname)
                print(coordinates_fname)

                np.save(self.icosahedral_coordinates_path, self.coordinates)

                if os.path.isfile(self.output_save_dir / stats_fname) and self.global_normalization:
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

            self.data_path = self.save_data_into_disk(self.raw_data.astype("float32"), fname, self.output_save_dir)
            # print("In cmip6, just saved the data.")

            # print("In cmip6, just about to copy the data to slurm.")
            # self.copy_to_slurm(self.input_path)
            self.copy_to_slurm(self.data_path)
            # print("In cmip6, just copied the data to slurm.")

            self.Data = self.norm_data.astype("float32")

        # plot_species(self.Data[:, :, 0, :, :], self.coordinates, variables, "../../TEST_REPO", "before_causal")
        # self.Data = self._reload_data(self.data_path)

        # Now X and Y is ready for getitem
        # print("CMIP6 shape", self.Data.shape)
        self.length = self.Data.shape[0]

    def __getitem__(self, index):
        return self.Data[index]

    def __len__(self):
        return self.length


def visualize_remapped_data(
    raw_spatial_data,
    coordinates,
    save_path="remapped_scatter.png",
    timestep=0,
    var_names=None,
    scenario_names=None,
):
    """
    Visualize spatial remapped data + global variables.

    Parameters
    ----------
    raw_spatial_data :
        shape = (num_scenarios, t, num_spatial_vars, npix)

    raw_global_data :
        shape = (num_scenarios, t, num_global_vars, 1, 1)

    coordinates :
        shape = (npix, 2)
        [:,0] = lon
        [:,1] = lat
    """

    num_scenarios, T, num_spatial_vars, npix = raw_spatial_data.shape

    lon = coordinates[:, 0]
    lat = coordinates[:, 1]

    # -------------------------------------------------------
    # global variables
    # -------------------------------------------------------
    num_global_vars = 0

    total_cols = num_spatial_vars + num_global_vars

    # -------------------------------------------------------
    # names
    # -------------------------------------------------------
    if var_names is None:
        var_names = [f"SpatialVar_{i}" for i in range(num_spatial_vars)]

    if scenario_names is None:
        scenario_names = [f"Scenario_{i}" for i in range(num_scenarios)]

    # -------------------------------------------------------
    # figure
    # -------------------------------------------------------
    fig, axes = plt.subplots(
        num_scenarios,
        total_cols,
        figsize=(4 * num_spatial_vars, 3.5 * num_scenarios),
        squeeze=False,
    )

    # -------------------------------------------------------
    # spatial variables
    # -------------------------------------------------------
    for s in range(num_scenarios):

        for v in range(num_spatial_vars):

            ax = axes[s, v]

            values = raw_spatial_data[s, timestep, v]

            scatter = ax.scatter(
                lon,
                lat,
                c=values,
                s=2,
                cmap="coolwarm",
            )

            ax.set_title(f"{scenario_names[s]}\n{var_names[v]}")

            ax.set_xlabel("Lon")
            ax.set_ylabel("Lat")

            plt.colorbar(scatter, ax=ax, shrink=0.7)

    plt.tight_layout()

    plt.savefig(save_path, bbox_inches="tight")

    plt.close()

    print(f"Saved visualization to: {save_path}")


class MultiscenarioDataset(ClimateDataset):

    def __init__(
        self,
        years,
        climate_model: str = "NorESM2-LM",
        historical_years: Union[int, str, None] = None,
        data_dir: Optional[str] = "Climateset_DATA",
        scenarios: Optional[List[str]] = ["ssp126", "ssp245", "ssp370", "ssp585"],
        variables: Optional[List[str]] = ["tas", "ts", "co2mass", "mmrbc", "so2", "ch4global"],
        ensembles: Optional[List[str]] = ["r1i1p1f1"],
        channels_last: bool = True,
        seq_to_seq: bool = True,
        mode: str = "train",
        output_save_dir: str = "",
        map_to_healpix: bool = True,
        reload_climate_set_data: bool = True,
        seq_len: int = 12,
        lat: int = 96,
        lon: int = 144,
        global_normalization: bool = True,
        seasonality_removal: bool = True,
        icosahedral_coordinates_path: str = "../../mappings/vertex_lonlat_mapping.npy",
        *args,
        **kwargs,
    ):
        scenarios = [s for s in scenarios if s != "historical"]
        self.root_dir = Path(data_dir)
        self.output_save_dir = Path(output_save_dir)
        self.scenarios = scenarios
        self.variables = variables
        self.spatial_variables = [v for v in variables if v in SPATIAL_VARIABLES]
        self.target_variables = [v for v in variables if v in TARGET_VARIABLES]
        self.global_forcing_variables = [v for v in variables if v in GLOBAL_FORCING_VARIABLES]
        self.spatial_forcing_variables = [v for v in variables if v in SPATIAL_FORCING_VARIABLES]
        print("spatial_forcing_variables", self.spatial_forcing_variables)
        print("global_forcing_variables", self.global_forcing_variables)
        print("spatial_variables", self.spatial_variables)
        self.ensemble = ensembles[0]
        self.seq_len = seq_len
        self.mode = mode
        self.map_to_healpix = map_to_healpix
        self.lat = lat
        self.lon = lon
        self.global_normalization = global_normalization
        self.seasonality_removal = seasonality_removal
        self.icosahedral_coordinates_path = icosahedral_coordinates_path
        fname_kwargs_ssp = dict(
            climate_model=climate_model,
            num_ensembles=len(ensembles),
            variables=variables,
            scenarios=scenarios,
            channels_last=channels_last,
            seq_to_seq=seq_to_seq,
        )
        self.fname_kwargs_ssp = fname_kwargs_ssp

        fname_kwargs_hist = dict(
            climate_model=climate_model,
            num_ensembles=len(ensembles),
            years=f"{years[0]}-{years[-1]}",
            historical_years=f"{historical_years[0]}-{historical_years[-1]}",
            variables=variables,
            scenarios=["historical"],
            channels_last=channels_last,
            seq_to_seq=seq_to_seq,
        )
        self.fname_kwargs_hist = fname_kwargs_hist
        # fname_kwargs_picontrol = dict(
        #     climate_model=climate_model,
        #     num_ensembles=len(ensembles),
        #     variables=self.target_variables,
        #     scenarios=["piControl"],
        #     channels_last=channels_last,
        #     seq_to_seq=seq_to_seq,
        # )

        # ------- Load raw data, save, and then remap ------------- "

        files_by_kind_ssp = self._collect_files("ScenarioMIP", scenarios, self.variables)  # dict {var:[paths]}

        # Load SSP spatial and global data
        raw_spatial_data_ssp = self._load_spatial_data(
            files_by_kind_ssp["spatial"], self.spatial_variables
        )  # (num_scenarios, t, variables, h, w); t=1032
        raw_global_data_ssp = self._load_global_data(files_by_kind_ssp["global"])  # (num_scenarios, t, variables, 1, 1)

        fname_spatial_ssp, coordinates_fname = self.get_save_name_from_kwargs(
            mode=mode, file="spatial", kwargs=self.fname_kwargs_ssp
        )
        fname_global_ssp, _ = self.get_save_name_from_kwargs(mode=mode, file="global", kwargs=self.fname_kwargs_ssp)

        self.spatial_ssp_data_path = self.save_data_into_disk(
            raw_spatial_data_ssp.astype("float32"), fname_spatial_ssp, self.output_save_dir
        )
        self.global_ssp_data_path = self.save_data_into_disk(
            raw_global_data_ssp.astype("float32"), fname_global_ssp, self.output_save_dir
        )

        # Load piControl target data
        # will only be used as data sesonality removal purpose
        files_by_kind_picontrol = self._collect_files("CMIP", ["piControl"], self.target_variables)
        raw_target_data_picontrol = self._load_spatial_data(files_by_kind_picontrol["spatial"], self.target_variables)

        if self.mode in ["train", "train+val"]:
            # Load historical spatial and global data
            files_by_kind_hist = self._collect_files("CMIP", ["historical"], self.variables)
            # historical data is only used for training and validation
            raw_spatial_data_hist = self._load_spatial_data(
                files_by_kind_hist["spatial"], self.spatial_variables
            )  # (1, T, variables, h, w); T=1980
            raw_global_data_hist = self._load_global_data(
                files_by_kind_hist["global"]
            )  # (num_scenarios, T, variables, 1, 1)

            fname_spatial_hist, _ = self.get_save_name_from_kwargs(
                mode=mode, file="spatial", kwargs=self.fname_kwargs_hist
            )
            fname_global_hist, _ = self.get_save_name_from_kwargs(
                mode=mode, file="global", kwargs=self.fname_kwargs_hist
            )

            self.spatial_hist_data_path = self.save_data_into_disk(
                raw_spatial_data_hist.astype("float32"), fname_spatial_hist, self.output_save_dir
            )
            self.global_hist_data_path = self.save_data_into_disk(
                raw_global_data_hist.astype("float32"), fname_global_hist, self.output_save_dir
            )

        lon, lat = self.load_coordinates_into_mem(files_by_kind_ssp["spatial"][self.spatial_variables[0]])

        print(f"coordinates_fname {coordinates_fname}")

        if self.map_to_healpix:
            print("remapping to healpix")
            raw_spatial_data_ssp, latitudes_new, longitudes_new = remap_reg_to_healpix(
                raw_spatial_data_ssp, lon, lat
            )  # (num_scenarios, t, variables, npix)
            raw_global_data_ssp = raw_global_data_ssp.squeeze(-1)  # (1, T, variables, 1)
            self.coordinates = np.c_[longitudes_new, latitudes_new]

            raw_target_data_picontrol, _, _ = remap_reg_to_healpix(raw_target_data_picontrol, lon, lat)

            if self.mode in ["train", "train+val"]:
                raw_spatial_data_hist, _, _ = remap_reg_to_healpix(raw_spatial_data_hist, lon, lat)
                raw_global_data_hist = raw_global_data_hist.squeeze(-1)  # (1, T, variables, 1)

            # pi_time   = np.arange(1600, 2101, 1/12)
            # hist_time = np.arange(1850, 2015, 1/12)
            # ssp_time  = np.arange(2015, 2101, 1/12)
            # v = 0
            # plt.figure(figsize=(8, 4))
            # save_path= f"line-plot_raw_{self.spatial_variables[v]}"
            # plt.plot(pi_time, raw_target_data_picontrol[0,:,v].mean((-1)).reshape(-1), alpha=0.5, label=f"piControl {self.spatial_variables[v]}")
            # plt.plot(hist_time, raw_spatial_data_hist[0,:,v].mean((-1)).reshape(-1), alpha=0.5, label=f"historical {self.spatial_variables[v]}")
            # for s in range(len(self.scenarios)):
            #     # flat = np.r_[self.raw_target_data_picontrol[0,:,v].mean((-1)).reshape(-1), self.raw_spatial_data_historical[0,:,v].mean((-1)).reshape(-1) , self.raw_spatial_data_ssp[s,:,v].mean((-1)).reshape(-1)]
            #     plt.plot(ssp_time, raw_spatial_data_ssp[s,:,v].mean((-1)).reshape(-1), alpha=0.5, label=f"{self.scenarios[s]} {self.spatial_variables[v]}")
            # plt.title(f"Spatial Data (raw)")
            # plt.legend()
            # plt.xlabel("Time")
            # plt.ylabel("Value")

            # plt.tight_layout()
            # plt.savefig(save_path, dpi=300)
            # plt.close()

        else:
            self.coordinates = self.load_coordinates_into_mem(files_by_kind_ssp["spatial"][self.spatial_variables[0]])
            self.coordinates = np.meshgrid(self.coordinates[0], self.coordinates[1])
            self.coordinates = np.c_[self.coordinates[1].flatten(), self.coordinates[0].flatten()]

        # Separate spatial data into target data and spatial forcings
        target_idx = [self.spatial_variables.index(target_variable) for target_variable in self.target_variables]
        # print("target_idx", target_idx) 0
        spatial_forcing_idx = [i for i in range(len(self.spatial_variables)) if i not in target_idx]
        # print("spatial_forcing_idx", spatial_forcing_idx) [1,2]

        # ============================================
        # First, try seasonality removal
        # ============================================

        raw_target_data_ssp = raw_spatial_data_ssp[:, :, target_idx]
        raw_spatial_forcing_data_ssp = raw_spatial_data_ssp[:, :, spatial_forcing_idx]
        raw_global_forcing_data_ssp = raw_global_data_ssp
        if self.mode in ["train", "train+val"]:
            raw_target_data_hist = raw_spatial_data_hist[:, :, target_idx]
            raw_spatial_forcing_data_hist = raw_spatial_data_hist[:, :, spatial_forcing_idx]
            raw_global_forcing_data_hist = raw_global_data_hist

        if self.seasonality_removal:
            # ----------------------------------------
            # TRAIN / TRAIN+VAL
            # compute seasonality from picontrol
            if self.mode in ["train", "train+val"]:
                target_data_picontrol, remove_season_stats = self.remove_seasonality(
                    raw_target_data_picontrol, months_per_year=self.seq_len
                )

                self.write_dataset_statistics("remove_season_stats", remove_season_stats)

            # ----------------------------------------
            # TEST
            else:
                remove_season_stats = self.load_dataset_statistics(
                    "remove_season_stats.npy", mode=self.mode, mips="cmip"
                )
            # apply SAME climatology to SSP and historical target variable
            if self.mode in ["train", "train+val"]:
                target_data_hist, _ = self.remove_seasonality(
                    raw_target_data_hist, months_per_year=self.seq_len, remove_season_stats=remove_season_stats
                )
            target_data_ssp, _ = self.remove_seasonality(
                raw_target_data_ssp, months_per_year=self.seq_len, remove_season_stats=remove_season_stats
            )

        # ============================================
        # no seasonality removal
        # ============================================

        else:
            if self.mode in ["train", "train+val"]:
                target_data_hist = raw_target_data_hist
            target_data_ssp = raw_target_data_ssp
        # skip the seasonality removal for forcing data. We want to preserve the seasonality of forcings
        if self.mode in ["train", "train+val"]:
            global_forcing_data_hist = raw_global_forcing_data_hist
            spatial_forcing_data_hist = raw_spatial_forcing_data_hist
        global_forcing_data_ssp = raw_global_forcing_data_ssp
        spatial_forcing_data_ssp = raw_spatial_forcing_data_ssp

        # for v in range(len(self.target_variables)):
        #     plt.figure(figsize=(8, 4))
        #     save_path= f"line-plot_deseaconlized_{self.target_variables[v]}-target"
        #     plt.plot(pi_time, target_data_picontrol[0,:,v].mean((-1)).reshape(-1), alpha=0.5, label=f"piControl {self.target_variables[v]}")
        #     plt.plot(hist_time, target_data_hist[0,:,v].mean((-1)).reshape(-1), alpha=0.5, label=f"historical {self.target_variables[v]}")
        #     for s in range(len(self.scenarios)):
        #         # flat = np.r_[self.norm_target_data_picontrol[0,:,v].mean((-1)).reshape(-1), self.norm_spatial_data_historical[0,:,v].mean((-1)).reshape(-1), self.norm_spatial_data_ssp[s,:,v].mean((-1)).reshape(-1)]
        #         # plt.plot(flat, alpha=0.5, label=f"{self.scenarios[s]} {self.spatial_variables[v]}")
        #         plt.plot(ssp_time, target_data_ssp[s,:,v].mean((-1)).reshape(-1), alpha=0.5, label=f"{self.scenarios[s]} {self.target_variables[v]}")
        #     plt.title(f"Spatial Data (deseasonlized")
        #     plt.legend()
        #     plt.xlabel("Time")
        #     plt.ylabel("Value")

        #     plt.tight_layout()
        #     plt.savefig(save_path, dpi=150)
        #     plt.close()

        # ------- data normalization ------------- "
        stats_fname, _ = self.get_save_name_from_kwargs(
            mode=self.mode,
            file="statistics",
            kwargs=self.fname_kwargs_hist,
        )
        if self.global_normalization:
            # ----------------------------------------
            # TRAIN / TRAIN+VAL
            # compute stats from historical
            if self.mode in ["train", "train+val"]:
                self.norm_spatial_forcing_data_hist, spatial_forcing_stats = self._apply_normalization(
                    spatial_forcing_data_hist
                )
                self.norm_global_forcing_data_hist, global_forcing_stats = self._apply_normalization(
                    global_forcing_data_hist
                )
                self.norm_target_data_hist, target_stats = self._apply_normalization(target_data_hist)
                stats = {}
                stats["spatial_forcing"] = spatial_forcing_stats
                stats["global_forcing"] = global_forcing_stats  # (variables, 1, 1, 1)
                stats["target"] = target_stats
                self.write_dataset_statistics(stats_fname, stats)
            # ----------------------------------------
            # TEST
            else:
                stats = self.load_dataset_statistics(stats_fname, mode=self.mode, mips="cmip")
            # apply SAME statistics to SSP
            self.norm_spatial_forcing_data_ssp, _ = self._apply_normalization(
                spatial_forcing_data_ssp, stats["spatial_forcing"]
            )
            self.norm_global_forcing_data_ssp, _ = self._apply_normalization(
                global_forcing_data_ssp, stats["global_forcing"]
            )
            self.norm_target_data_ssp, _ = self._apply_normalization(target_data_ssp, stats["target"])
            # picontrol
            # norm_target_data_picontrol, _= self._apply_normalization(
            #     target_data_picontrol,
            #     stats["target"]
            # )

        # ============================================
        # no global normalization
        # ============================================
        else:
            if self.mode in ["train", "train+val"]:
                self.norm_spatial_forcing_data_hist = spatial_forcing_data_hist.astype("float32")
                self.norm_global_forcing_data_hist = global_forcing_data_hist.astype("float32")
                self.norm_target_data_hist = target_data_hist.astype("float32")

            self.norm_spatial_forcing_data_ssp = spatial_forcing_data_ssp.astype("float32")
            self.norm_global_forcing_data_ssp = global_forcing_data_ssp.astype("float32")
            # norm_target_data_picontrol = target_data_picontrol
            self.norm_target_data_ssp = target_data_ssp.astype("float32")
        # if self.mode in ["train", "train+val"]:
        #     for v in range(len(self.global_forcing_variables)):
        #         plt.figure(figsize=(8, 4))
        #         save_path= f"line-plot-normalized_{self.global_forcing_variables[v]}"
        #         plt.plot(hist_time,self.norm_global_forcing_data_hist[0,:,v].mean((-1)).reshape(-1), alpha=0.5, label=f"historical {self.global_forcing_variables[v]}")
        #         for s in range(len(self.scenarios)):
        #             plt.plot(ssp_time,self.norm_global_forcing_data_ssp[s,:,v,0].reshape(-1) , alpha=0.5, label=f"{self.scenarios[s]} {self.global_forcing_variables[v]}")
        #         plt.title(f"Global Data (normalized)")
        #         plt.legend()
        #         plt.xlabel("Time")
        #         plt.ylabel("Value")

        #         plt.tight_layout()
        #         plt.savefig(save_path, dpi=150)
        #         plt.close()

        # for v in range(len(self.target_variables)):
        #     plt.figure(figsize=(8, 4))
        #     save_path= f"line-plot_deseaconlized-normalized_{self.target_variables[v]}-target"
        #     plt.plot(pi_time, norm_target_data_picontrol[0,:,v].mean((-1)).reshape(-1), alpha=0.5, label=f"piControl {self.target_variables[v]}")
        #     plt.plot(hist_time,self.norm_target_data_hist[0,:,v].mean((-1)).reshape(-1), alpha=0.5, label=f"historical {self.target_variables[v]}")
        #     for s in range(len(self.scenarios)):
        #         plt.plot(ssp_time,self.norm_target_data_ssp[s,:,v].mean((-1)).reshape(-1), alpha=0.5, label=f"{self.scenarios[s]} {self.target_variables[v]}")
        #     plt.title(f"Spatial Data (deseasonlized-normalized)")
        #     plt.legend()
        #     plt.xlabel("Time")
        #     plt.ylabel("Value")

        #     plt.tight_layout()
        #     plt.savefig(save_path, dpi=150)
        #     plt.close()

        # for v in range(len(self.spatial_forcing_variables)):
        #     plt.figure(figsize=(8, 4))
        #     save_path= f"line-plot_normalized_{self.spatial_forcing_variables[v]}"
        #     plt.plot(hist_time,self.norm_spatial_forcing_data_hist[0,:,v].mean((-1)).reshape(-1), alpha=0.5, label=f"historical {self.spatial_forcing_variables[v]}")
        #     for s in range(len(self.scenarios)):
        #         plt.plot(ssp_time,self.norm_spatial_forcing_data_ssp[s,:,v].mean((-1)).reshape(-1), alpha=0.5, label=f"{self.scenarios[s]} {self.spatial_forcing_variables[v]}")
        #     plt.title(f"Spatial Data (normalized)")
        #     plt.legend()
        #     plt.xlabel("Time")
        #     plt.ylabel("Value")

        #     plt.tight_layout()
        #     plt.savefig(save_path, dpi=150)
        #     plt.close()

    def _table_for_variable(self, variable: str) -> str:
        return "AERmon" if variable in AERMON_VARIABLES else "Amon"

    def _grid_for_variable(self, variable: str) -> str:
        return "gm" if variable in GM_VARIABLES else "gn"

    def _load_time_data(self, files_by_kind):
        reference_files = None

        if len(files_by_kind["spatial"]) > 0:
            first_var = self.spatial_variables[0]
            reference_files = files_by_kind["spatial"][first_var]
        elif len(files_by_kind["global"]) > 0:
            first_var = self.global_forcing_variables[0]
            reference_files = files_by_kind["global"][first_var]
        else:
            raise ValueError("No files available to load time coordinates.")

        scenario_time_arrays = []

        for scenario_files in reference_files:
            ds = xr.open_mfdataset(scenario_files, chunks={"time": 120})
            ds = ds.drop_dims("bnds", errors="ignore")
            ds = ds.compute()

            if "time" not in ds.coords:
                raise KeyError(f"No time coordinate found in files: {scenario_files[:3]}")

            scenario_time_arrays.append(ds["time"].values)

        time_values = np.concatenate(scenario_time_arrays, axis=0)

        if time_values.shape[0] % self.seq_len != 0:
            raise ValueError(f"Time length {time_values.shape[0]} is not divisible by seq_len={self.seq_len}")
        return time_values.reshape(-1, self.seq_len)  #

    def _collect_files(self, mip, domains, variables):
        files = {
            "spatial": {},
            "global": {},
        }

        for variable in variables:
            if variable in SPATIAL_VARIABLES:
                kind = "spatial"
            elif variable in GLOBAL_FORCING_VARIABLES:
                kind = "global"
            else:
                raise ValueError(f"Unknown variable kind for {variable}")

            files[kind][variable] = []

            for scenario in domains:
                table = self._table_for_variable(variable)
                grid = self._grid_for_variable(variable)

                var_dir = self.root_dir / mip / "NCC/NorESM2-LM" / scenario / self.ensemble / table / variable / grid

                # find all nc files (version name **)
                scenario_files = sorted(glob.glob(f"{var_dir}/**/*.nc", recursive=True))
                #  if no nc files, then try to serach grib files
                if len(scenario_files) == 0:
                    scenario_files = sorted(glob.glob(f"{var_dir}/**/*.grib", recursive=True))
                # if neither nc nor grid were found return error
                if len(scenario_files) == 0:
                    raise FileNotFoundError(
                        f"No files found for variable={variable}, scenario={scenario}, path={var_dir}"
                    )
                files[kind][variable].append(scenario_files)

        return files

    def _load_spatial_data(self, files_by_variable, spatial_variables):
        if len(files_by_variable) == 0:
            return None

        variable_arrays = []

        for variable in spatial_variables:
            scenario_arrays = []

            for scenario_files in files_by_variable[variable]:
                ds = xr.open_mfdataset(scenario_files, chunks={"time": 120})
                ds = ds.drop_dims("bnds", errors="ignore")

                if "plev" in ds.sizes:
                    ds = ds.mean("plev")
                elif "lev" in ds.sizes:
                    ds = ds.mean("lev")

                ds = ds.compute()

                arr = ds[variable].to_numpy()

                if arr.ndim == 4:
                    arr = np.squeeze(arr)

                if arr.ndim != 3:
                    raise ValueError(
                        f"Spatial variable {variable} should have shape " f"(time, lat, lon), got {arr.shape}"
                    )
                # add dim0: scenario dim
                scenario_arrays.append(arr[None])

            variable_array = np.concatenate(scenario_arrays, axis=0)
            # print("variable_array",variable_array.shape)
            variable_arrays.append(variable_array)
        data = np.stack(
            variable_arrays, axis=0
        )  # (num_variables, num_scenarios, time, lat, lon) # (3, 2, 1032, 96, 144)
        data = data.transpose((1, 2, 0, 3, 4))  # (num_scenarios, time, num_variables, lat, lon)

        return data.astype("float32")

    def _load_global_data(self, files_by_variable):
        if len(files_by_variable) == 0:
            return None

        variable_arrays = []

        for variable in self.global_forcing_variables:
            scenario_arrays = []

            for scenario_files in files_by_variable[variable]:
                ds = xr.open_mfdataset(scenario_files, chunks={"time": 120})
                ds = ds.drop_dims("bnds", errors="ignore")
                ds = ds.compute()

                arr = ds[variable]
                reduce_dims = [dim for dim in arr.dims if dim != "time"]
                if len(reduce_dims) > 0:
                    arr = arr.mean(dim=reduce_dims, skipna=True)

                arr = arr.to_numpy()
                arr = np.squeeze(arr)

                if arr.ndim != 1:
                    raise ValueError(f"Global variable {variable} should have shape " f"(time,), got {arr.shape}")
                # add dim0: scenario dim
                scenario_arrays.append(arr[None])

            variable_array = np.concatenate(scenario_arrays, axis=0)
            variable_arrays.append(variable_array)

        data = np.stack(variable_arrays, axis=0)

        data = data.transpose((1, 2, 0))[..., None, None]  # (scenarios, time, global_variables, 1, 1)

        return data.astype("float32")

    def remove_seasonality(
        self, x, months_per_year=12, remove_season_stats=None
    ):  # overriten the seasonality removal functions
        """
        x: (num_scenarios, t, var, lat, lon) or (num_scenarios, t, var, npix)
        returns deseasonalized x + climatology
        """
        n_scenarios, T, n_var = x.shape[:3]
        spatial_shape = x.shape[3:]

        assert T % months_per_year == 0, f"t must be multiple of {months_per_year}"

        years = T // months_per_year

        # (scenario, year, month, var, ...)
        x_reshaped = x.reshape(n_scenarios, years, months_per_year, n_var, *spatial_shape)

        # (1, 1, month, var, ...)
        if remove_season_stats is None:
            mean = np.nanmean(x_reshaped, axis=(0, 1), keepdims=True)
            std = np.nanstd(x_reshaped, axis=(0, 1), keepdims=True)
            remove_season_stats = {"mean": mean, "std": std}
            print(f"Compute the seasonality of shape{mean.shape} from {n_scenarios} domain.")
        else:
            mean = remove_season_stats["mean"]
            std = remove_season_stats["std"]
            print(f"Using the exsisting seasonality for {n_scenarios} domain")
        print("Just about to return the data after removing seasonality.")

        x_deseasonal = x_reshaped - mean

        # back to original shape
        x_deseasonal = x_deseasonal.reshape(x.shape)

        return x_deseasonal.astype("float32"), remove_season_stats

    def _apply_normalization(self, data, stats=None):
        if stats is None:
            mean, std = self.get_mean_std(data)
            stats = {
                "mean": mean,
                "std": std,
            }
            norm_data = self.normalize_data(
                data,
                stats,
            )
        else:
            norm_data = self.normalize_data(
                data,
                stats,
            )
        return norm_data.astype("float32"), stats

    def get_causal_data(
        self,
        data,
        tau,
        future_timesteps,
        mode,
        num_months_aggregated=1,
        ratio_train=None,
        interval_length=100,
        domain_names=None,
    ):
        """
        Constructs dataset for causal discovery model.

        Splits each scenario into training and validation sets, then generates overlapping sequences.
        """
        # scenrios, years*months, vars, lon, lat)
        print(f"get_causal_data {mode} input data shape {data.shape} of domains {domain_names}")
        if isinstance(num_months_aggregated, (int, np.integer)) and num_months_aggregated > 1:
            data = self.aggregate_months(data, num_months_aggregated)
            # for each scenario in data, generate overlapping sequences
            if mode == "train" or mode == "train+val":
                # print("IN IF")
                x_train_list, y_train_list = [], []
                x_valid_list, y_valid_list = [], []

                for scenario in data:
                    idx_train, idx_valid = self.split_data_by_interval(scenario, tau, ratio_train, interval_length)
                    # np.random.shuffle(idx_train)
                    # np.random.shuffle(idx_valid)

                    x_train, y_train = self.get_overlapping_sequences(scenario, idx_train, tau, future_timesteps)
                    x_train_list.extend(x_train)
                    y_train_list.extend(y_train)
                    x_valid, y_valid = self.get_overlapping_sequences(scenario, idx_valid, tau, future_timesteps)
                    x_valid_list.extend(x_valid)
                    y_valid_list.extend(y_valid)

                train_x, train_y = np.stack(x_train_list), np.stack(y_train_list)
                if ratio_train == 1:
                    valid_x, valid_y = np.array(x_valid_list), np.array(y_valid_list)
                else:
                    valid_x, valid_y = np.stack(x_valid_list), np.stack(y_valid_list)

                train = train_x, train_y
                valid = valid_x, valid_y

                return train, valid
            else:
                x_test_list, y_test_list = [], []
                for scenario in data:
                    idx_test = np.arange(tau, scenario.shape[0])
                    x_test, y_test = self.get_overlapping_sequences(scenario, idx_test, tau, future_timesteps)
                    x_test_list.extend(x_test)
                    y_test_list.extend(y_test)

                test_x, test_y = np.stack(x_test_list), np.stack(y_test_list)
                # test_y = np.expand_dims(test_y, axis=1)

                test = test_x, test_y
                # test = self.normalize_data(test_x, stats_x), self.normalize_data(test_y, stats_y)

                return test

        else:
            # "here"
            # TODO create this function and use it -> put it inside the data creation...
            # data = self.create_multi_res_data(data, num_months_aggregated)

            # for each scenario in data, generate overlapping sequences
            if mode == "train" or mode == "train+val":
                x_train_list, y_train_list, domain_train_list = [], [], []
                x_valid_list, y_valid_list, domain_valid_list = [], [], []

                for scenario, name in zip(data, domain_names):
                    idx_train, idx_valid = self.split_data_by_interval(scenario, tau, ratio_train, interval_length)

                    x_train, y_train = self.get_overlapping_sequences(scenario, idx_train, tau, future_timesteps)
                    x_train_list.extend(x_train)  # len(x_train) 775
                    y_train_list.extend(y_train)

                    domain_train_list.extend([name] * len(x_train))

                    x_valid, y_valid = self.get_overlapping_sequences(scenario, idx_valid, tau, future_timesteps)
                    x_valid_list.extend(x_valid)
                    y_valid_list.extend(y_valid)  # len(x_valid)  100
                    print("x_valid len", len(x_valid))
                    domain_valid_list.extend([name] * len(x_valid))

                train_x, train_y = np.stack(x_train_list), np.stack(
                    y_train_list
                )  # train_x (775*num_scenarios, 5, 2, 3072)
                train_domain = np.array(domain_train_list)  # train_domain (775*num_scenarios,)

                if ratio_train == 1:
                    valid_x, valid_y = np.array(x_valid_list), np.array(y_valid_list)
                else:
                    valid_x, valid_y = np.stack(x_valid_list), np.stack(y_valid_list)
                valid_domain = np.array(domain_valid_list)
                train = train_x, train_y, train_domain
                valid = valid_x, valid_y, valid_domain

                return train, valid
            else:
                x_test_list, y_test_list, domain_test_list = [], [], []
                for scenario, name in zip(data, domain_names):
                    idx_test = np.arange(tau, scenario.shape[0])
                    x_test, y_test = self.get_overlapping_sequences(scenario, idx_test, tau, future_timesteps)
                    x_test_list.extend(x_test)
                    y_test_list.extend(y_test)
                    domain_test_list.extend([name] * len(x_test))

                test_x, test_y = np.stack(x_test_list), np.stack(y_test_list)
                test_domain = np.array(domain_test_list)
                # test_y = np.expand_dims(test_y, axis=1)
                test = (test_x, test_y), test_domain
                # test_x (1027, 5, 1, 3072) test_y (1027, 1, 1, 1, 3072)
                return test


if __name__ == "__main__":
    sample_dataset = MultiscenarioDataset(
        years="1600-1602",
        historical_years="1894-1896",
        data_dir="/network/scratch/j/julien.boussard/ESMValTool/climate_data/CMIP6",
        scenarios=["ssp126", "ssp245", "ssp370", "historical"],
        variables=["ts", "co2mass", "mmrbc", "so2", "ch4global"],
        mode="train",
        output_save_dir="/home/mila/s/shanz/scratch/data/healpix_data_reducedim_4_mutiscario_test",
        reload_climate_set_data=False,
        seasonality_removal=True,
        seq_len=12,  # one .nc file is of length 12 months
        lat=96,
        lon=144,
    )
