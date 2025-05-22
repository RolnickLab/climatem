from climatem.data_loader.climate_dataset import ClimateDataset
from pathlib import Path
import xarray as xr
import numpy as np
from typing import List, Optional


class SingleFileDailyDataset(ClimateDataset):
    def __init__(
        self,
        file_path: str,
        variables: List[str],  # <-- now supports multiple variables
        seq_len: int = 365,
        mode: str = "train",
        channels_last: bool = True,
        seq_to_seq: bool = True,
        global_normalization: bool = True,
        seasonality_removal: bool = True,
        reload_climate_set_data: bool = True,
        output_save_dir: str = "./",
        lat: int = 166,
        lon: int = 244,
        *args,
        **kwargs,
    ):
        super().__init__()

        self.file_path = Path(file_path)
        self.variables = variables
        self.seq_len = seq_len
        self.mode = mode
        self.channels_last = channels_last
        self.seq_to_seq = seq_to_seq
        self.global_normalization = global_normalization
        self.seasonality_removal = seasonality_removal
        self.reload_climate_set_data = reload_climate_set_data
        self.output_save_dir = Path(output_save_dir)
        self.lat = lat
        self.lon = lon

        # Extract available years from the file
        ds = xr.open_dataset(self.file_path).compute()
        times = ds["time"].to_index()
        self.years = sorted(set(times.year))
        self.length = len(self.years)  # number of sequences = number of years

        # Load data into memory using ClimateDataset method
        self.raw_data = self.load_into_mem(
            paths=[str(self.file_path)],
            variables=self.variables,
            channels_last=self.channels_last,
            seq_to_seq=self.seq_to_seq,
            get_years=self.years,
        )

        # Store total flattened spatial dim
        self.d_x = self.input_var_offsets[-1]

        # Apply optional normalization and seasonality removal
        if self.global_normalization:
            stat1, stat2 = self.get_dataset_statistics(self.raw_data, self.mode, mips="cmip6")
            stats = {"mean": stat1, "std": stat2}
            self.norm_data = self.normalize_data(self.raw_data, stats)
        else:
            self.norm_data = self.raw_data

        if self.seasonality_removal:
            self.norm_data = self.remove_seasonality(self.norm_data)

        self.Data = self.norm_data

    def __getitem__(self, index):
        return self.Data[index]
