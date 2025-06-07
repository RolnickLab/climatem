import glob
import os
import warnings
from pathlib import Path
from typing import List, Optional, Union

from climatem.utils import get_logger

from .teleconnections_dataset import TeleconnectionsDataset

warnings.filterwarnings("ignore", category=FutureWarning, module="cfgrib")

log = get_logger()


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
            files_per_var = []
            for var in variables:
                pattern_grib = f"{self.output_save_dir}/*{var}*.nc"
                grib_files = glob.glob(pattern_grib, recursive=True)
                files_per_var.append(grib_files)

            self.raw_data, self.input_var_shapes, self.input_var_offsets, self.coordinates = self.load_into_mem(
                files_per_var,
                variables,
                channels_last,
                seq_to_seq,
            )

            if self.mode in ["train", "train+val"]:
                stats_fname, coordinates_fname = self.get_save_name_from_kwargs(
                    mode=mode, file="statistics", kwargs=self.fname_kwargs
                )
                if os.path.isfile(stats_fname) and self.global_normalization:
                    stats = self.load_dataset_statistics(stats_fname, mode=self.mode, mips="cmip6")
                    self.norm_data = self.normalize_data(self.raw_data, stats)
                elif self.global_normalization:
                    stat1, stat2 = self.get_dataset_statistics(self.raw_data, self.mode, mips="cmip6")
                    stats = {"mean": stat1, "std": stat2}
                    self.norm_data = self.normalize_data(self.raw_data, stats)
                    self.write_dataset_statistics(stats_fname, stats)
                else:
                    self.norm_data = self.raw_data
                if self.seasonality_removal:
                    self.norm_data = self.remove_seasonality(self.norm_data)
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

            self.data_path = self.save_data_into_disk(self.raw_data, fname, self.output_save_dir)
            self.copy_to_slurm(self.data_path)
            self.Data = self.norm_data

        self.length = self.Data.shape[0]

    def __getitem__(self, index):
        return self.Data[index]
