import glob
import os
import sys
import warnings
from pathlib import Path
from typing import List, Optional, Union

import numpy as np

from climatem.utils import get_logger

from .teleconnections_dataset import TeleconnectionsDataset

warnings.filterwarnings("ignore", category=FutureWarning, module="cfgrib")

log = get_logger()


def exit_with_error():
    print("This doesn't work for now.")
    sys.exit(1)  # Exit with a non-zero status code to indicate an error


class CHIRPSDataset(TeleconnectionsDataset):
    def __init__(
        self,
        years: Union[int, str],
        historical_years: Union[int, str],
        data_dir: Optional[str] = "",
        climate_model: str = "",
        num_ensembles: int = 0,
        scenarios: List[str] = [""],
        variables: List[str] = ["lsp"],
        mode: str = "train",
        output_save_dir: str = "",
        reload_climate_set_data: bool = True,
        channels_last: bool = True,
        seq_to_seq: bool = True,
        global_normalization: bool = True,
        seasonality_removal: bool = True,
        seq_len: int = 365,
        lat: int = 96,
        lon: int = 144,
        icosahedral_coordinates_path: str = "",
        season: str = "all",
        *args,
        **kwargs,
    ):
        self.mode = mode
        self.output_save_dir = Path(output_save_dir)
        self.reload_climate_set_data = reload_climate_set_data
        self.root_dir = Path(data_dir)
        self.variables = variables
        self.global_normalization = global_normalization
        self.seasonality_removal = seasonality_removal
        self.seq_len = seq_len
        self.lon = lon
        self.lat = lat
        self.icosahedral_coordinates_path = icosahedral_coordinates_path
        self.input_var_shapes = {}
        self.input_var_offsets = [0]
        self.season = season

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

        if isinstance(climate_model, str):
            self.root_dir = self.root_dir / climate_model
        else:
            log.warn("Data loader not yet implemented for multiple climate models.")
            raise NotImplementedError

        if num_ensembles == 1:
            ensemble = os.listdir(self.root_dir)
            self.ensemble_dirs = [self.root_dir / ensemble[0]]
        elif num_ensembles == 0:
            self.ensemble_dirs = [self.root_dir]
        else:
            ensembles = os.listdir(self.root_dir)
            self.ensemble_dirs = [self.root_dir / ensemble for ensemble in ensembles]

        fname, coordinates_fname = self.get_save_name_from_kwargs(mode=mode, file="target", kwargs=self.fname_kwargs)

        if (self.output_save_dir / fname).is_file() and self.reload_climate_set_data:
            print("for now this doesn't work...")
            exit_with_error()
            self.data_path = self.output_save_dir / fname
            self.raw_data = self._reload_data(self.data_path)
            self.coordinates = self.load_dataset_coordinates(coordinates_fname, mode=self.mode, mips="cmip6")
            if self.global_normalization:
                stats_fname, _ = self.get_save_name_from_kwargs(mode=mode, file="statistics", kwargs=self.fname_kwargs)
                stats = self.load_dataset_statistics(stats_fname, mode=self.mode, mips="cmip6")
                self.Data = self.normalize_data(self.raw_data, stats)
            else:
                self.Data = self.raw_data
            if self.seasonality_removal:
                self.Data = self.remove_seasonality(self.Data)
        else:
            print("NOT RELOADING!!")
            files_per_var = []
            for var in variables:
                subdir_nc = f"{self.output_save_dir}/{var}/**/*.nc"
                subdir_grib = f"{self.output_save_dir}/{var}/**/*.grib*"
                matched_nc = sorted(glob.glob(subdir_nc, recursive=True))
                matched_grib = sorted(glob.glob(subdir_grib, recursive=True))

                if not matched_nc and not matched_grib:
                    flat_nc = f"{self.output_save_dir}/*{var}*.nc"
                    print(f"flat_nc - not matched_nc : {flat_nc}")
                    flat_grib = f"{self.output_save_dir}/*{var}*.grib*"
                    print(f"flat_grib - not matched_grib : {flat_nc}")
                    matched_nc = sorted(glob.glob(flat_nc))
                    matched_grib = sorted(glob.glob(flat_grib))

                matched_grib = [f for f in matched_grib if "000366" not in f]

                all_matched = matched_nc if matched_nc else matched_grib

                print(f"all_matched : {all_matched}")

                if not all_matched:
                    raise FileNotFoundError(f"No .nc or .grib files found for variable '{var}'")

                files_per_var.append(all_matched)
            (
                self.Data,
                self.input_var_shapes,
                self.input_var_offsets,
                self.coordinates,
                self.new_lat,
                self.new_lon,
            ) = self.load_into_mem(
                files_per_var,
                variables,
                channels_last,
                seq_to_seq,
                upscaling_factor=2,
                season=self.season,
                mode=self.mode,
                global_normalization=self.global_normalization,
                seasonality_removal=self.seasonality_removal,
            )

            print(f"np.nanmax(self.raw_data) {np.nanmax(self.Data)}")

            # if self.mode in ["train", "train+val"]:
            #     stats_fname, coordinates_fname = self.get_save_name_from_kwargs(
            #         mode=mode, file="statistics", kwargs=self.fname_kwargs
            #     )
            #     # if os.path.isfile(stats_fname) and self.global_normalization:
            #     #     stats = self.load_dataset_statistics(stats_fname, mode=self.mode, mips="cmip6")
            #     #     self.norm_data = self.normalize_data(self.raw_data, stats)
            #     # elif self.global_normalization:
            #     if self.global_normalization:
            #         stat1, stat2 = self.get_dataset_statistics(self.raw_data, self.mode, mips="cmip6")
            #         stats = {"mean": stat1, "std": stat2}
            #         self.norm_data = self.normalize_data(self.raw_data, stats)
            #         print(f"AFTER global_normalization np.nanmax(self.norm_data) {np.nanmax(self.norm_data)}")
            #         self.write_dataset_statistics(stats_fname, stats)
            #     else:
            #         self.norm_data = self.raw_data
            #     if self.seasonality_removal:
            #         self.norm_data = self.remove_seasonality(self.norm_data, self.idx_season)
            #         print(f"AFTER seasonality_removal np.nanmax(self.norm_data) {np.nanmax(self.norm_data)}")
            # elif self.mode == "test":
            #     if self.global_normalization:
            #         stats_fname, coordinates_fname = self.get_save_name_from_kwargs(
            #             mode="train+val", file="statistics", kwargs=fname_kwargs
            #         )
            #         stats = self.load_dataset_statistics(stats_fname, mode=self.mode, mips="cmip6")
            #         self.norm_data = self.normalize_data(self.raw_data, stats)
            #     else:
            #         self.norm_data = self.raw_data
            #     if self.seasonality_removal:
            #         self.norm_data = self.remove_seasonality(self.norm_data, self.idx_season)
            # self.data_path = self.save_data_into_disk(self.raw_data, fname, self.output_save_dir)
            # self.copy_to_slurm(self.data_path)
            # self.Data = self.norm_data

        self.length = self.Data.shape[0]

    def __getitem__(self, index):
        return self.Data[index]
