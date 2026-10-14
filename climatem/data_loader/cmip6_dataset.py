# NOTE: Un-verified implementation for get lagged global forcing! (you have to choose which function you would like)

import glob
import os
from pathlib import Path
from typing import List, Optional, Union

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
from climatem.data_loader.healpix_remapping import remap_reg_to_healpix
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
            )

            lon, lat = self.load_coordinates_into_mem(files_per_var)

            if self.map_to_healpix:
                print("remapping to healpix")
                self.raw_data, latitudes_new, longitudes_new = remap_reg_to_healpix(self.raw_data, lon, lat)
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


class MultiscenarioDataset(ClimateDataset):

    def __init__(
        self,
        years,
        climate_model: str = "NorESM2-LM",
        historical_years: Union[int, str, None] = None,
        data_dir: Optional[str] = "Climateset_DATA",
        scenarios: Optional[List[str]] = ["ssp126", "ssp245", "ssp370", "ssp585", "abrupt-4xCO2"],
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
        use_x4 = "abrupt-4xCO2" in scenarios
        ssp_scenarios = [s for s in scenarios if s != "abrupt-4xCO2"]
        has_ssp = len(ssp_scenarios) > 0
        if not has_ssp and not use_x4:
            raise ValueError("At least one scenario must be provided.")

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
        self.reload_climate_set_data = reload_climate_set_data
        self.icosahedral_coordinates_path = icosahedral_coordinates_path
        fname_kwargs_ssp = dict(
            climate_model=climate_model,
            num_ensembles=len(ensembles),
            variables=variables,
            scenarios=ssp_scenarios,
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
        fname_kwargs_picontrol = dict(
            climate_model=climate_model,
            num_ensembles=len(ensembles),
            variables=self.target_variables,
            scenarios=["piControl"],
            channels_last=channels_last,
            seq_to_seq=seq_to_seq,
        )
        fname_kwargs_x4 = dict(
            climate_model="NorESM2-MM",
            num_ensembles=len(ensembles),
            variables=variables,
            scenarios=["abrupt-4xCO2"],
            years="1-86",
            aerosol_years="1841-1926",
            target_grid="NorESM2-LM",
            channels_last=channels_last,
            seq_to_seq=seq_to_seq,
        )

        # ------- Load raw data, save, and then remap ------------- "

        files_by_kind_ssp = None
        # Load SSP spatial and global data
        if has_ssp:
            files_by_kind_ssp = self._collect_files("ScenarioMIP", ssp_scenarios, self.variables)  # dict {var:[paths]}

            fname_spatial_ssp, _ = self.get_save_name_from_kwargs(
                mode=mode, file="spatial", kwargs=self.fname_kwargs_ssp
            )
            fname_global_ssp, _ = self.get_save_name_from_kwargs(mode=mode, file="global", kwargs=self.fname_kwargs_ssp)

            self.spatial_ssp_data_path = self.output_save_dir / fname_spatial_ssp
            self.global_ssp_data_path = self.output_save_dir / fname_global_ssp

            raw_spatial_data_ssp = self._load_or_build_cached_array(
                self.spatial_ssp_data_path,
                lambda: self._load_spatial_data(files_by_kind_ssp["spatial"], self.spatial_variables),
                fname_spatial_ssp,
            )  # (num_scenarios, T, num_var, 96, 144)

            raw_global_data_ssp = self._load_or_build_cached_array(
                self.global_ssp_data_path,
                lambda: self._load_global_data(files_by_kind_ssp["global"]),
                fname_global_ssp,
            )

        # Load piControl target data
        # will only be used as data sesonality removal and coordinates loading purpose
        fname_target_picontrol, coordinates_fname = self.get_save_name_from_kwargs(
            mode=mode, file="target_picontrol", kwargs=fname_kwargs_picontrol
        )

        self.target_picontrol_data_path = self.output_save_dir / fname_target_picontrol

        files_by_kind_picontrol = self._collect_files("CMIP", ["piControl"], self.target_variables)

        raw_target_data_picontrol = self._load_or_build_cached_array(
            self.target_picontrol_data_path,
            lambda: self._load_spatial_data(files_by_kind_picontrol["spatial"], self.target_variables),
            fname_target_picontrol,
        )
        # We also need global forcing in test
        fname_global_hist, _ = self.get_save_name_from_kwargs(mode=mode, file="global", kwargs=self.fname_kwargs_hist)
        self.global_hist_data_path = self.output_save_dir / fname_global_hist

        files_by_kind_hist = self._collect_files("CMIP", ["historical"], self.variables)
        raw_global_data_hist = self._load_or_build_cached_array(
            self.global_hist_data_path,
            lambda: self._load_global_data(files_by_kind_hist["global"]),
            fname_global_hist,
        )  # (num_scenarios, T, variables, 1, 1)
        # Load historical spatial and global data
        # historical data is only used for training and validation
        if self.mode in ["train", "train+val"]:
            fname_spatial_hist, _ = self.get_save_name_from_kwargs(
                mode=mode, file="spatial", kwargs=self.fname_kwargs_hist
            )
            self.spatial_hist_data_path = self.output_save_dir / fname_spatial_hist
            raw_spatial_data_hist = self._load_or_build_cached_array(
                self.spatial_hist_data_path,
                lambda: self._load_spatial_data(files_by_kind_hist["spatial"], self.spatial_variables),
                fname_spatial_hist,
            )  # (1, T, variables, h, w); T=1980
        # We use the coordinates from piControl
        lon, lat = self.load_coordinates_into_mem(files_by_kind_picontrol["spatial"][self.target_variables[0]])

        print(f"coordinates_fname {coordinates_fname}")

        if use_x4:
            files_by_kind_x4 = self._collect_files_extreme_case(
                "CMIP",
                ["abrupt-4xCO2"],
                self.variables,
            )

            fname_spatial_x4, _ = self.get_save_name_from_kwargs(mode=mode, file="spatial_x4", kwargs=fname_kwargs_x4)

            fname_global_x4, _ = self.get_save_name_from_kwargs(mode=mode, file="global_x4", kwargs=fname_kwargs_x4)

            spatial_x4_path = self.output_save_dir / fname_spatial_x4
            global_x4_path = self.output_save_dir / fname_global_x4

            raw_spatial_data_x4 = self._load_or_build_cached_array(
                spatial_x4_path,
                lambda: self._load_spatial_data_extreme(
                    files_by_kind_x4["spatial"],
                    self.spatial_variables,
                    lm_coords=(lon, lat),
                ),
                fname_spatial_x4,
            )

            raw_global_data_x4 = self._load_or_build_cached_array(
                global_x4_path,
                lambda: self._load_global_data_extreme(
                    files_by_kind_x4["global"],
                ),
                fname_global_x4,
            )

            # Here we append the x4 case as a ssp scenario!
            if has_ssp:
                # SSP + X4
                raw_spatial_data_ssp = np.concatenate([raw_spatial_data_ssp, raw_spatial_data_x4], axis=0)
                raw_global_data_ssp = np.concatenate([raw_global_data_ssp, raw_global_data_x4], axis=0)
            else:
                # X4 only
                raw_spatial_data_ssp = raw_spatial_data_x4
                raw_global_data_ssp = raw_global_data_x4
            print(
                f"Append abrupt-x4CO2 to the exsiting scenarios! Now spatial data has a shape of {raw_spatial_data_ssp.shape}, global data of the shape {raw_global_data_ssp.shape}"
            )
        if self.map_to_healpix:
            print("remapping to healpix")
            raw_spatial_data_ssp, latitudes_new, longitudes_new = remap_reg_to_healpix(
                raw_spatial_data_ssp, lon, lat
            )  # (num_scenarios, t, variables, npix)
            raw_global_data_ssp = raw_global_data_ssp.squeeze(-1)  # (1, T, variables, 1)
            self.coordinates = np.c_[longitudes_new, latitudes_new]

            raw_target_data_picontrol, _, _ = remap_reg_to_healpix(raw_target_data_picontrol, lon, lat)

            raw_global_data_hist = raw_global_data_hist.squeeze(-1)  # (1, T, variables, 1)
            if self.mode in ["train", "train+val"]:
                raw_spatial_data_hist, _, _ = remap_reg_to_healpix(raw_spatial_data_hist, lon, lat)

        else:
            self.coordinates = self.load_coordinates_into_mem(
                files_by_kind_picontrol["spatial"][self.target_variables[0]]
            )
            self.coordinates = np.meshgrid(self.coordinates[0], self.coordinates[1])
            self.coordinates = np.c_[self.coordinates[1].flatten(), self.coordinates[0].flatten()]

        self.write_dataset_statistics(coordinates_fname, self.coordinates)
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

        raw_global_forcing_data_hist = raw_global_data_hist
        if self.mode in ["train", "train+val"]:
            raw_target_data_hist = raw_spatial_data_hist[:, :, target_idx]
            raw_spatial_forcing_data_hist = raw_spatial_data_hist[:, :, spatial_forcing_idx]

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
            spatial_forcing_data_hist = raw_spatial_forcing_data_hist
        global_forcing_data_hist = raw_global_forcing_data_hist
        global_forcing_data_ssp = raw_global_forcing_data_ssp
        spatial_forcing_data_ssp = raw_spatial_forcing_data_ssp

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
                stats["target"] = target_stats  # mean 0.12984121  std 1.3174365
                self.write_dataset_statistics(stats_fname, stats)
            # ----------------------------------------
            # TEST
            else:
                stats = self.load_dataset_statistics(stats_fname, mode=self.mode, mips="cmip")
                self.norm_global_forcing_data_hist, _ = self._apply_normalization(
                    global_forcing_data_hist, stats["global_forcing"]
                )

            # apply SAME statistics to SSP
            self.norm_spatial_forcing_data_ssp, _ = self._apply_normalization(
                spatial_forcing_data_ssp, stats["spatial_forcing"]
            )
            # co2:1.6155315e+15, 1.5112757e+14;  ch4 1.1897872e+03; 3.3693771e+02
            self.norm_global_forcing_data_ssp, _ = self._apply_normalization(
                global_forcing_data_ssp, stats["global_forcing"]
            )
            self.norm_target_data_ssp, _ = self._apply_normalization(target_data_ssp, stats["target"])

        # ============================================
        # no global normalization
        # ============================================
        else:
            if self.mode in ["train", "train+val"]:
                self.norm_spatial_forcing_data_hist = spatial_forcing_data_hist.astype("float32")
                self.norm_target_data_hist = target_data_hist.astype("float32")

            self.norm_global_forcing_data_hist = global_forcing_data_hist.astype("float32")
            self.norm_spatial_forcing_data_ssp = spatial_forcing_data_ssp.astype("float32")
            self.norm_global_forcing_data_ssp = global_forcing_data_ssp.astype("float32")
            self.norm_target_data_ssp = target_data_ssp.astype("float32")

    def _table_for_variable(self, variable: str) -> str:
        return "AERmon" if variable in AERMON_VARIABLES else "Amon"

    def _grid_for_variable(self, variable: str) -> str:
        return "gm" if variable in GM_VARIABLES else "gn"

    def _load_or_build_cached_array(self, path, build_fn, fname):
        path = Path(path)

        if self.reload_climate_set_data and path.exists():
            print(f"Loading cached ClimateSet array: {path}")
            data = self._reload_data(path)
            return data.astype("float32")

        print(f"Building ClimateSet array and saving to: {path}")
        data = build_fn().astype("float32")
        self.save_data_into_disk(data, fname, self.output_save_dir)
        return data

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

                var_dir = (
                    self.root_dir / mip / "NCC" / "NorESM2-LM" / scenario / self.ensemble / table / variable / grid
                )

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

    def _collect_files_extreme_case(self, mip, domains, variables):
        # x4 CO2 data, the ts, co2mass, ch4global are from abrupt-4xCO2 folder (150 year), but the mmrbc and so2 are taken from the piControl (1850-2000).
        files = {"spatial": {}, "global": {}}
        model = "NorESM2-MM"

        scenario_override = {
            "mmrbc": "piControl",
            "so2": "piControl",
        }
        model_override = {
            "mmrbc": "NorESM2-LM",
            "so2": "NorESM2-LM",
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
                data_scenario = scenario_override.get(variable, scenario)
                data_model = model_override.get(variable, model)

                table = self._table_for_variable(variable)
                grid = self._grid_for_variable(variable)

                var_dir = (
                    self.root_dir / mip / "NCC" / data_model / data_scenario / self.ensemble / table / variable / grid
                )

                scenario_files = sorted(glob.glob(f"{var_dir}/**/*.nc", recursive=True))

                if not scenario_files:
                    scenario_files = sorted(glob.glob(f"{var_dir}/**/*.grib", recursive=True))

                if not scenario_files:
                    raise FileNotFoundError(
                        f"No files found for variable={variable}, " f"scenario={data_scenario}, path={var_dir}"
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

    def _load_spatial_data_extreme(
        self,
        files_by_variable,
        spatial_variables,
        lm_coords,
        n_months=86 * 12,
    ):
        """
        Load abrupt-4xCO2 spatial data.

        ts:abrupt-4xCO2, NorESM2-MM, first 86 years

        mmrbc / so2: piControl, NorESM2-LM, 1841-1926

        Returns
        -------
        data:(1, 1020, n_variables, npix)
        """
        lm_lon, lm_lat = lm_coords
        variable_arrays = []

        for variable in spatial_variables:
            scenario_files = files_by_variable[variable][0]

            ds = xr.open_mfdataset(
                scenario_files,
                combine="by_coords",
                chunks={"time": 120},
            )
            # Inpterpolate the coordinates if they are not from the same resolution
            # TS is from MM, so we have to first downsample it to the LM resolution
            if ds["lat"].size != len(lm_lat) or ds["lon"].size != len(lm_lon):
                ds = ds.interp(lat=lm_lat, lon=lm_lon, method="linear")
            ds = ds.drop_dims("bnds", errors="ignore")
            # ----------------------------
            # time alignment
            # ----------------------------
            if variable in {"mmrbc", "so2"}:
                year = ds.time.dt.year
                ds = ds.sel(time=(year >= 1841) & (year <= 1926))

            else:  # abrupt-4xCO2: experiment years 1-86
                ds = ds.isel(time=slice(0, n_months))

            if ds.sizes["time"] != n_months:
                raise ValueError(f"{variable}: expected {n_months} months, " f"got {ds.sizes['time']}")

            # ----------------------------
            # vertical reduction
            # ----------------------------
            if "plev" in ds.sizes:
                ds = ds.mean("plev")
            elif "lev" in ds.sizes:
                ds = ds.mean("lev")

            da = ds[variable].compute()
            arr = da.to_numpy()

            if arr.ndim != 3:
                raise ValueError(f"{variable}: expected (time, lat, lon), " f"got {arr.shape}")
            variable_arrays.append(arr)

        # (T, variable, npix)
        data = np.stack(variable_arrays, axis=1)
        # add scenario dimension
        data = data[None]
        return data.astype("float32")

    def _load_global_data_extreme(
        self,
        files_by_variable,
        n_months=86 * 12,
    ):
        """Load global forcings for abrupt-4xCO2."""

        variable_arrays = []

        for variable in self.global_forcing_variables:

            scenario_files = files_by_variable[variable][0]
            ds = xr.open_mfdataset(
                scenario_files,
                combine="by_coords",
                chunks={"time": 120},
            )
            ds = ds.drop_dims("bnds", errors="ignore")

            # first 85 experiment years
            ds = ds.isel(time=slice(0, n_months))
            da = ds[variable]

            reduce_dims = [d for d in da.dims if d != "time"]
            if reduce_dims:
                da = da.mean(dim=reduce_dims, skipna=True)

            arr = np.squeeze(da.compute().to_numpy())

            if arr.shape[0] != n_months:
                raise ValueError(f"{variable}: expected {n_months} months, " f"got {arr.shape[0]}")

            variable_arrays.append(arr)

        # T, variable
        data = np.stack(variable_arrays, axis=1)

        # scenario, T, variable, 1, 1
        data = data[None, :, :, None, None]

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


class MultiScaleMultiscenarioDataset(ClimateDataset):

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
        global_lag=240,
        *args,
        **kwargs,
    ):
        # SSP scenarios only. Historical is loaded separately and used as
        # training data / prefix for long global-forcing histories.
        scenarios = [s for s in scenarios if s != "historical"]
        use_x4 = "abrupt-4xCO2" in scenarios
        ssp_scenarios = [s for s in scenarios if s != "abrupt-4xCO2"]
        has_ssp = len(ssp_scenarios) > 0
        if not has_ssp and not use_x4:
            raise ValueError("At least one scenario must be provided.")

        # Global forcing uses a 240-month lookback INCLUDING the current month:
        # [t-240, ..., t] -> 241 stored monthly values.
        self.global_lookback = global_lag

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
        self.reload_climate_set_data = reload_climate_set_data
        self.icosahedral_coordinates_path = icosahedral_coordinates_path
        fname_kwargs_ssp = dict(
            climate_model=climate_model,
            num_ensembles=len(ensembles),
            variables=variables,
            scenarios=ssp_scenarios,
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
        fname_kwargs_picontrol = dict(
            climate_model=climate_model,
            num_ensembles=len(ensembles),
            variables=self.target_variables,
            scenarios=["piControl"],
            channels_last=channels_last,
            seq_to_seq=seq_to_seq,
        )

        fname_kwargs_x4 = dict(
            climate_model="NorESM2-MM",
            num_ensembles=len(ensembles),
            variables=variables,
            scenarios=["abrupt-4xCO2"],
            years="1-86",
            aerosol_years="1841-1926",
            target_grid="NorESM2-LM",
            channels_last=channels_last,
            seq_to_seq=seq_to_seq,
        )

        # ------- Load raw data, save, and then remap ------------- "

        # Load SSP spatial and global data
        files_by_kind_ssp = None
        if has_ssp:
            files_by_kind_ssp = self._collect_files("ScenarioMIP", ssp_scenarios, self.variables)  # dict {var:[paths]}

            fname_spatial_ssp, _ = self.get_save_name_from_kwargs(
                mode=mode, file="spatial", kwargs=self.fname_kwargs_ssp
            )
            fname_global_ssp, _ = self.get_save_name_from_kwargs(mode=mode, file="global", kwargs=self.fname_kwargs_ssp)

            self.spatial_ssp_data_path = self.output_save_dir / fname_spatial_ssp
            self.global_ssp_data_path = self.output_save_dir / fname_global_ssp

            raw_spatial_data_ssp = self._load_or_build_cached_array(
                self.spatial_ssp_data_path,
                lambda: self._load_spatial_data(files_by_kind_ssp["spatial"], self.spatial_variables),
                fname_spatial_ssp,
            )  # (num_scenarios, T, num_var, 96, 144)

            raw_global_data_ssp = self._load_or_build_cached_array(
                self.global_ssp_data_path,
                lambda: self._load_global_data(files_by_kind_ssp["global"]),
                fname_global_ssp,
            )

        # Load piControl target data
        # will only be used as data sesonality removal and coordinates loading purpose
        fname_target_picontrol, coordinates_fname = self.get_save_name_from_kwargs(
            mode=mode, file="target_picontrol", kwargs=fname_kwargs_picontrol
        )

        self.target_picontrol_data_path = self.output_save_dir / fname_target_picontrol

        files_by_kind_picontrol = self._collect_files("CMIP", ["piControl"], self.target_variables)

        raw_target_data_picontrol = self._load_or_build_cached_array(
            self.target_picontrol_data_path,
            lambda: self._load_spatial_data(files_by_kind_picontrol["spatial"], self.target_variables),
            fname_target_picontrol,
        )
        # We also need global forcing in test
        fname_global_hist, _ = self.get_save_name_from_kwargs(mode=mode, file="global", kwargs=self.fname_kwargs_hist)
        self.global_hist_data_path = self.output_save_dir / fname_global_hist

        files_by_kind_hist = self._collect_files("CMIP", ["historical"], self.variables)
        raw_global_data_hist = self._load_or_build_cached_array(
            self.global_hist_data_path,
            lambda: self._load_global_data(files_by_kind_hist["global"]),
            fname_global_hist,
        )  # (num_scenarios, T, variables, 1, 1)
        # Load historical spatial and global data
        # historical data is only used for training and validation
        if self.mode in ["train", "train+val"]:
            fname_spatial_hist, _ = self.get_save_name_from_kwargs(
                mode=mode, file="spatial", kwargs=self.fname_kwargs_hist
            )
            self.spatial_hist_data_path = self.output_save_dir / fname_spatial_hist
            raw_spatial_data_hist = self._load_or_build_cached_array(
                self.spatial_hist_data_path,
                lambda: self._load_spatial_data(files_by_kind_hist["spatial"], self.spatial_variables),
                fname_spatial_hist,
            )  # (1, T, variables, h, w); T=1980
        # We use the coordinates from piControl
        lon, lat = self.load_coordinates_into_mem(files_by_kind_picontrol["spatial"][self.target_variables[0]])

        print(f"coordinates_fname {coordinates_fname}")
        if use_x4:
            files_by_kind_x4 = self._collect_files_extreme_case(
                "CMIP",
                ["abrupt-4xCO2"],
                self.variables,
            )
            fname_spatial_x4, _ = self.get_save_name_from_kwargs(mode=mode, file="spatial_x4", kwargs=fname_kwargs_x4)
            fname_global_x4, _ = self.get_save_name_from_kwargs(mode=mode, file="global_x4", kwargs=fname_kwargs_x4)

            spatial_x4_path = self.output_save_dir / fname_spatial_x4
            global_x4_path = self.output_save_dir / fname_global_x4

            raw_spatial_data_x4 = self._load_or_build_cached_array(
                spatial_x4_path,
                lambda: self._load_spatial_data_extreme(
                    files_by_kind_x4["spatial"],
                    self.spatial_variables,
                    lm_coords=(lon, lat),
                ),
                fname_spatial_x4,
            )

            raw_global_data_x4 = self._load_or_build_cached_array(
                global_x4_path,
                lambda: self._load_global_data_extreme(
                    files_by_kind_x4["global"],
                ),
                fname_global_x4,
            )

            # Here we append the x4 case as a ssp scenario!
            if has_ssp:
                # SSP + X4
                raw_spatial_data_ssp = np.concatenate(
                    [raw_spatial_data_ssp, raw_spatial_data_x4],
                    axis=0,
                )
                raw_global_data_ssp = np.concatenate(
                    [raw_global_data_ssp, raw_global_data_x4],
                    axis=0,
                )
            else:
                # X4 only
                raw_spatial_data_ssp = raw_spatial_data_x4
                raw_global_data_ssp = raw_global_data_x4
            print(
                f"Append abrupt-x4CO2 to the exsiting scenarios! Now spatial data has a shape of {raw_spatial_data_ssp.shape}, global data of the shape {raw_global_data_ssp.shape}"
            )
            # afterwards, the abrupt-x4CO2 data will be simply taken as one type of SSP data
        if self.map_to_healpix:
            print("remapping to healpix")
            raw_spatial_data_ssp, latitudes_new, longitudes_new = remap_reg_to_healpix(
                raw_spatial_data_ssp, lon, lat
            )  # (num_scenarios, t, variables, npix)
            raw_global_data_ssp = raw_global_data_ssp.squeeze(-1)  # (1, T, variables, 1)
            self.coordinates = np.c_[longitudes_new, latitudes_new]

            raw_target_data_picontrol, _, _ = remap_reg_to_healpix(raw_target_data_picontrol, lon, lat)

            raw_global_data_hist = raw_global_data_hist.squeeze(-1)  # (1, T, variables, 1)
            if self.mode in ["train", "train+val"]:
                raw_spatial_data_hist, _, _ = remap_reg_to_healpix(raw_spatial_data_hist, lon, lat)

        else:
            self.coordinates = self.load_coordinates_into_mem(
                files_by_kind_picontrol["spatial"][self.target_variables[0]]
            )
            self.coordinates = np.meshgrid(self.coordinates[0], self.coordinates[1])
            self.coordinates = np.c_[self.coordinates[1].flatten(), self.coordinates[0].flatten()]

        self.write_dataset_statistics(coordinates_fname, self.coordinates)
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

        raw_global_forcing_data_hist = raw_global_data_hist
        if self.mode in ["train", "train+val"]:
            raw_target_data_hist = raw_spatial_data_hist[:, :, target_idx]
            raw_spatial_forcing_data_hist = raw_spatial_data_hist[:, :, spatial_forcing_idx]

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
            spatial_forcing_data_hist = raw_spatial_forcing_data_hist
        global_forcing_data_hist = raw_global_forcing_data_hist
        global_forcing_data_ssp = raw_global_forcing_data_ssp
        spatial_forcing_data_ssp = raw_spatial_forcing_data_ssp

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
                stats["target"] = target_stats  # mean 0.12984121  std 1.3174365
                self.write_dataset_statistics(stats_fname, stats)
            # ----------------------------------------
            # TEST
            else:
                stats = self.load_dataset_statistics(stats_fname, mode=self.mode, mips="cmip")
                self.norm_global_forcing_data_hist, _ = self._apply_normalization(
                    global_forcing_data_hist, stats["global_forcing"]
                )

            # apply SAME statistics to SSP
            self.norm_spatial_forcing_data_ssp, _ = self._apply_normalization(
                spatial_forcing_data_ssp, stats["spatial_forcing"]
            )
            # co2:1.6155315e+15, 1.5112757e+14;  ch4 1.1897872e+03; 3.3693771e+02
            self.norm_global_forcing_data_ssp, _ = self._apply_normalization(
                global_forcing_data_ssp, stats["global_forcing"]
            )
            self.norm_target_data_ssp, _ = self._apply_normalization(target_data_ssp, stats["target"])

        # ============================================
        # no global normalization
        # ============================================
        else:
            if self.mode in ["train", "train+val"]:
                self.norm_spatial_forcing_data_hist = spatial_forcing_data_hist.astype("float32")
                self.norm_target_data_hist = target_data_hist.astype("float32")
            self.norm_global_forcing_data_hist = global_forcing_data_hist.astype("float32")
            self.norm_spatial_forcing_data_ssp = spatial_forcing_data_ssp.astype("float32")
            self.norm_global_forcing_data_ssp = global_forcing_data_ssp.astype("float32")
            self.norm_target_data_ssp = target_data_ssp.astype("float32")

    def _table_for_variable(self, variable: str) -> str:
        return "AERmon" if variable in AERMON_VARIABLES else "Amon"

    def _grid_for_variable(self, variable: str) -> str:
        return "gm" if variable in GM_VARIABLES else "gn"

    def _load_or_build_cached_array(self, path, build_fn, fname):
        path = Path(path)

        if self.reload_climate_set_data and path.exists():
            print(f"Loading cached ClimateSet array: {path}")
            data = self._reload_data(path)
            return data.astype("float32")

        print(f"Building ClimateSet array and saving to: {path}")
        data = build_fn().astype("float32")
        self.save_data_into_disk(data, fname, self.output_save_dir)
        return data

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

                var_dir = (
                    self.root_dir / mip / "NCC" / "NorESM2-LM" / scenario / self.ensemble / table / variable / grid
                )

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

    def _collect_files_extreme_case(self, mip, domains, variables):
        # x4 CO2 data, the ts, co2mass, ch4global are from abrupt-4xCO2 folder (150 year), but the mmrbc and so2 are taken from the piControl (1850-2000).
        files = {"spatial": {}, "global": {}}
        model = "NorESM2-MM"

        scenario_override = {
            "mmrbc": "piControl",
            "so2": "piControl",
        }
        model_override = {
            "mmrbc": "NorESM2-LM",
            "so2": "NorESM2-LM",
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

                data_scenario = scenario_override.get(variable, scenario)
                data_model = model_override.get(variable, model)

                table = self._table_for_variable(variable)
                grid = self._grid_for_variable(variable)

                var_dir = (
                    self.root_dir / mip / "NCC" / data_model / data_scenario / self.ensemble / table / variable / grid
                )

                scenario_files = sorted(glob.glob(f"{var_dir}/**/*.nc", recursive=True))

                if not scenario_files:
                    scenario_files = sorted(glob.glob(f"{var_dir}/**/*.grib", recursive=True))

                if not scenario_files:
                    raise FileNotFoundError(
                        f"No files found for variable={variable}, " f"scenario={data_scenario}, path={var_dir}"
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

    def _load_spatial_data_extreme(
        self,
        files_by_variable,
        spatial_variables,
        lm_coords,
        n_months=86 * 12,
    ):
        """
        Load abrupt-4xCO2 spatial data.

        ts:abrupt-4xCO2, NorESM2-MM, first 86 years

        mmrbc / so2: piControl, NorESM2-LM, 1841-1926

        Returns
        -------
        data:(1, 1020, n_variables, npix)
        """
        lm_lon, lm_lat = lm_coords
        variable_arrays = []

        for variable in spatial_variables:
            scenario_files = files_by_variable[variable][0]

            ds = xr.open_mfdataset(
                scenario_files,
                combine="by_coords",
                chunks={"time": 120},
            )
            # Inpterpolate the coordinates if they are not from the same resolution
            # TS is from MM, so we have to first downsample it to the LM resolution
            if ds["lat"].size != len(lm_lat) or ds["lon"].size != len(lm_lon):
                ds = ds.interp(lat=lm_lat, lon=lm_lon, method="linear")
            ds = ds.drop_dims("bnds", errors="ignore")
            # ----------------------------
            # time alignment
            # ----------------------------
            if variable in {"mmrbc", "so2"}:
                year = ds.time.dt.year
                ds = ds.sel(time=(year >= 1841) & (year <= 1926))

            else:  # abrupt-4xCO2: experiment years 1-86
                ds = ds.isel(time=slice(0, n_months))

            if ds.sizes["time"] != n_months:
                raise ValueError(f"{variable}: expected {n_months} months, " f"got {ds.sizes['time']}")
            # ----------------------------
            # vertical reduction
            # ----------------------------
            if "plev" in ds.sizes:
                ds = ds.mean("plev")
            elif "lev" in ds.sizes:
                ds = ds.mean("lev")

            da = ds[variable].compute()
            arr = da.to_numpy()

            if arr.ndim != 3:
                raise ValueError(f"{variable}: expected (time, lat, lon), " f"got {arr.shape}")
            variable_arrays.append(arr)

        # (T, variable, npix)
        data = np.stack(variable_arrays, axis=1)
        # add scenario dimension
        data = data[None]
        return data.astype("float32")

    def _load_global_data_extreme(
        self,
        files_by_variable,
        n_months=86 * 12,
    ):
        """Load global forcings for abrupt-4xCO2."""

        variable_arrays = []

        for variable in self.global_forcing_variables:

            scenario_files = files_by_variable[variable][0]

            ds = xr.open_mfdataset(
                scenario_files,
                combine="by_coords",
                chunks={"time": 120},
            )

            ds = ds.drop_dims("bnds", errors="ignore")

            # first 85 experiment years
            ds = ds.isel(time=slice(0, n_months))
            da = ds[variable]
            reduce_dims = [d for d in da.dims if d != "time"]

            if reduce_dims:
                da = da.mean(
                    dim=reduce_dims,
                    skipna=True,
                )

            arr = np.squeeze(da.compute().to_numpy())

            if arr.shape[0] != n_months:
                raise ValueError(f"{variable}: expected {n_months} months, " f"got {arr.shape[0]}")
            variable_arrays.append(arr)

        # T, variable
        data = np.stack(variable_arrays, axis=1)

        # scenario, T, variable, 1, 1
        data = data[None, :, :, None, None]

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

    def _filter_indices_for_global_history(
        self,
        indices,
        global_lookback=240,
        prefix_length=0,
    ):
        """
        Keep only target-time indices for which [t-global_lookback, ..., t] is available.

        For historical data, prefix_length=0, so t >= 240 is required. For SSP data with a 240-month historical prefix,
        all normal target indices are valid.
        """
        indices = np.asarray(indices, dtype=np.int64)
        return indices[prefix_length + indices >= global_lookback]

    def get_global_forcing_windows(
        self,
        data,
        indices_by_scenario,
        historical_prefix=None,
        global_lookback=240,
    ):
        """
        Build long global-forcing histories aligned to externally supplied sample indices.

        Parameters
        ----------
        data : np.ndarray
            (scenario, time, global_var, ...)
        indices_by_scenario : list[np.ndarray]
            Current-time indices in the *original* data for each scenario.
            Use the same indices as the spatial/target branch.
        historical_prefix : np.ndarray or None
            Optional historical data immediately preceding SSP:
            (1, >=global_lookback, global_var, ...).
        global_lookback : int
            Maximum lag in months. 240 means each returned window contains
            241 values: [t-240, ..., t].

        Returns
        -------
        windows : np.ndarray
            (samples, global_lookback + 1, global_var, ...)
        """
        window_length = global_lookback + 1
        windows = []

        if historical_prefix is not None:
            data_ext = np.concatenate([historical_prefix, data], axis=1)
            offset = historical_prefix.shape[1]
        else:
            data_ext = data
            offset = 0

        for s, indices in enumerate(indices_by_scenario):
            for t in indices:
                t_ext = offset + int(t)
                start = t_ext - global_lookback
                end = t_ext + 1
                if start < 0:
                    continue
                window = data_ext[s, start:end]
                if window.shape[0] != window_length:
                    continue
                windows.append(window)

        if len(windows) == 0:
            return np.empty((0, window_length, *data.shape[2:]), dtype=data.dtype)

        return np.stack(windows).astype("float32")

    def get_multiscale_causal_data(
        self,
        data,
        global_data,
        spatial_tau,
        future_timesteps,
        mode,
        ratio_train=None,
        interval_length=100,
        domain_names=None,
        global_prefix=None,
        global_lookback=240,
    ):
        """
        Build spatial/target samples and aligned long global-forcing histories.

        This is the multiscale replacement for calling get_causal_data()
        independently for spatial and global data.

        Spatial branch
        --------------
        Uses spatial_tau (e.g. 5 months) through get_overlapping_sequences().

        Global branch
        -------------
        For every SAME target-time index t, builds
            [F_{t-240}, ..., F_t]
        with 241 monthly values.  CausalDataset.__getitem__ later samples
        the 6 response intervals from this long history.

        For SSP, pass the last 240 months of historical forcing as
        global_prefix so the first SSP samples can use pre-2015 history.

        Returns
        -------
        train / valid:
            (x, y, global_x, domain)

        test:
            ((x, y, global_x), domain)
        """
        print(
            f"get_multiscale_causal_data {mode}: "
            f"data={data.shape}, global_data={global_data.shape}, "
            f"spatial_tau={spatial_tau}, global_lookback={global_lookback}"
        )

        if data.shape[0] != global_data.shape[0]:
            raise ValueError(
                f"Scenario mismatch: data has {data.shape[0]} scenarios, " f"global_data has {global_data.shape[0]}."
            )

        if domain_names is None:
            domain_names = [str(i) for i in range(data.shape[0])]

        # Prepare SSP historical prefix if provided.
        # For training and validation, we will never touch abrupt-4xCO2.
        # Therefore, use the last global_lookback months of historical data as the prefix as a whole by repeat the dim 0 num_scenario times.
        prefix = None
        if mode in ["train", "train+val"]:
            prefix_length = 0
            if global_prefix is not None:
                # for ssp, use last global lag months in historical
                prefix = global_prefix[:, -global_lookback:]
                if prefix.shape[0] == 1 and global_data.shape[0] > 1:
                    prefix = np.repeat(prefix, global_data.shape[0], axis=0)
                if prefix.shape[0] != global_data.shape[0]:
                    raise ValueError(
                        f"global_prefix has {prefix.shape[0]} scenarios, " f"global_data has {global_data.shape[0]}."
                    )
                prefix_length = prefix.shape[1]
            x_train_list, y_train_list, global_train_list = [], [], []
            x_valid_list, y_valid_list, global_valid_list = [], [], []
            domain_train_list, domain_valid_list = [], []

            for s, (scenario, name) in enumerate(zip(data, domain_names)):
                idx_train, idx_valid = self.split_data_by_interval(
                    scenario,
                    spatial_tau,
                    ratio_train,
                    interval_length,
                )

                # IMPORTANT:
                # filter spatial/target indices BEFORE creating samples so that
                # x/y/global remain one-to-one aligned.
                idx_train = self._filter_indices_for_global_history(
                    idx_train,
                    global_lookback=global_lookback,
                    prefix_length=prefix_length,
                )
                idx_valid = self._filter_indices_for_global_history(
                    idx_valid,
                    global_lookback=global_lookback,
                    prefix_length=prefix_length,
                )

                x_train, y_train = self.get_overlapping_sequences(
                    scenario,
                    idx_train,
                    spatial_tau,
                    future_timesteps,
                )
                x_valid, y_valid = self.get_overlapping_sequences(
                    scenario,
                    idx_valid,
                    spatial_tau,
                    future_timesteps,
                )

                global_train = self.get_global_forcing_windows(
                    global_data[s : s + 1],
                    [idx_train],
                    historical_prefix=None if prefix is None else prefix[s : s + 1],
                    global_lookback=global_lookback,
                )
                global_valid = self.get_global_forcing_windows(
                    global_data[s : s + 1],
                    [idx_valid],
                    historical_prefix=None if prefix is None else prefix[s : s + 1],
                    global_lookback=global_lookback,
                )

                if not (len(x_train) == len(y_train) == len(global_train)):
                    raise RuntimeError(
                        f"Train alignment failed for {name}: "
                        f"x={len(x_train)}, y={len(y_train)}, "
                        f"global={len(global_train)}"
                    )

                if not (len(x_valid) == len(y_valid) == len(global_valid)):
                    raise RuntimeError(
                        f"Validation alignment failed for {name}: "
                        f"x={len(x_valid)}, y={len(y_valid)}, "
                        f"global={len(global_valid)}"
                    )

                x_train_list.extend(x_train)
                y_train_list.extend(y_train)
                global_train_list.extend(global_train)
                domain_train_list.extend([name] * len(x_train))

                x_valid_list.extend(x_valid)
                y_valid_list.extend(y_valid)
                global_valid_list.extend(global_valid)
                domain_valid_list.extend([name] * len(x_valid))

            train_x = np.stack(x_train_list)
            train_y = np.stack(y_train_list)
            train_global = np.stack(global_train_list)
            train_domain = np.asarray(domain_train_list)

            if len(x_valid_list) == 0:
                valid_x = np.asarray(x_valid_list)
                valid_y = np.asarray(y_valid_list)
                valid_global = np.asarray(global_valid_list)
            else:
                valid_x = np.stack(x_valid_list)
                valid_y = np.stack(y_valid_list)
                valid_global = np.stack(global_valid_list)

            valid_domain = np.asarray(domain_valid_list)

            train = (
                train_x,
                train_y,
                train_global,
                train_domain,
            )
            valid = (
                valid_x,
                valid_y,
                valid_global,
                valid_domain,
            )

            return train, valid

        # --------------------------
        # test
        # --------------------------
        x_test_list, y_test_list, global_test_list = [], [], []
        domain_test_list = []

        # The test set includes both SSP and abrupt-4xCO2 scenarios, which require different historical prefixes.
        # Therefore, select the prefix separately for each scenario inside the loop.
        for s, (scenario, name) in enumerate(zip(data, domain_names)):
            prefix_length = 0
            if global_prefix is not None:
                # for ssp, use last global lag months in historical
                if name == "abrupt-4xCO2":
                    print(f"prefix for {name} use the first {global_lookback} years")
                    prefix = global_prefix[:, :global_lookback]
                else:
                    print(f"prefix for {name} use the last {global_lookback} years")
                    prefix = global_prefix[:, -global_lookback:]
                prefix_length = prefix.shape[1]

            idx_test = np.arange(
                spatial_tau,
                scenario.shape[0],
            )

            idx_test = self._filter_indices_for_global_history(
                idx_test,
                global_lookback=global_lookback,
                prefix_length=prefix_length,
            )

            x_test, y_test = self.get_overlapping_sequences(
                scenario,
                idx_test,
                spatial_tau,
                future_timesteps,
            )

            global_test = self.get_global_forcing_windows(
                global_data[s : s + 1],
                [idx_test],
                historical_prefix=prefix,
                global_lookback=global_lookback,
            )

            if not (len(x_test) == len(y_test) == len(global_test)):
                raise RuntimeError(
                    f"Test alignment failed for {name}: "
                    f"x={len(x_test)}, y={len(y_test)}, "
                    f"global={len(global_test)}"
                )

            x_test_list.extend(x_test)
            y_test_list.extend(y_test)
            global_test_list.extend(global_test)
            domain_test_list.extend([name] * len(x_test))

        test_x = np.stack(x_test_list)
        test_y = np.stack(y_test_list)
        test_global = np.stack(global_test_list)
        test_domain = np.asarray(domain_test_list)

        return (test_x, test_y, test_global), test_domain


if __name__ == "__main__":
    sample_dataset = MultiScaleMultiscenarioDataset(
        years="1600-1602",
        historical_years="1894-1896",
        data_dir="/network/scratch/j/julien.boussard/ESMValTool/climate_data/CMIP6",
        scenarios=["ssp126", "ssp245", "historical"],
        variables=["ts", "co2mass", "ch4global", "mmrbc", "so2"],
        mode="train+val",
        output_save_dir="/home/mila/s/shanz/scratch/data/healpix_data_reducedim_4_mutiscario_test",
        reload_climate_set_data=True,
        seasonality_removal=True,
        seq_len=12,  # one .nc file is of length 12 months
        lat=96,
        lon=144,
        global_lag=240,
        global_normalization=True,
        channels_last=False,
    )
    sample_dataset.get_multiscale_causal_data(
        sample_dataset.norm_spatial_forcing_data_ssp,
        sample_dataset.norm_global_forcing_data_ssp,
        spatial_tau=5,
        future_timesteps=1,
        mode="train+val",
        ratio_train=0.9,
        interval_length=100,
        domain_names=sample_dataset.scenarios,
        global_prefix=sample_dataset.norm_global_forcing_data_hist,
        global_lookback=240,
    )
