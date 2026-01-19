import numpy as np
import pandas as pd
import torch
from pytorch_lightning import LightningDataModule
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from climatem.utils import get_logger

log = get_logger()

list_variables = ["TA", "VPD", "SWC", "CO2", "GPP"]
list_control = ["SW_IN", "WS"]


class FluxnetDataModule(LightningDataModule):
    def __init__(
        self,
        data_path: str = "FluxNET_data.csv",
        # rolling_mean: int = 1, IMplement when wanna do rolling mean of 90 / 365 days :)
        growing_season_filter_ndaysabove: int = 10,
        deseasonalize: bool = True,
        val_split: float = 0.1,
        batch_size: int = 128,
        num_workers: int = 0,
        pin_memory: bool = False,
    ):
        super().__init__()

        self.data_path = data_path
        # self.rolling_mean = rolling_mean
        self.growing_season_filter_ndaysabove = growing_season_filter_ndaysabove
        self.deseasonalize = deseasonalize
        self.val_split = val_split
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

    def setup(self, accelerator):
        """
        Load data.

        Set internal variables: self._data_train, self._data_val, self._data_test
        Set torch dataloaders
        """

        self.prepare_data()
        self.train_dataloader = self.train_dataloader(accelerator)
        self.val_dataloader = self.val_dataloader(accelerator)
        self.test_dataloader = self.test_dataloader(accelerator)

    def _shared_dataloader_kwargs(self) -> dict:
        shared_kwargs = dict(num_workers=int(self.num_workers), pin_memory=self.pin_memory)
        return shared_kwargs

    def prepare_data(self):
        """Download or reload data and transform it in numpy."""
        df = pd.read_csv(self.data_path)

        # quick-and-dirty growing-season filter: 10day mean above 0.5 quantile
        df = df[df["GPP"].rolling(window=self.growing_season_filter_ndaysabove).mean() > df["GPP"].quantile(0.5)]

        if self.deseasonalize:
            # #%% remove annual mean and deseasonalize
            df_deseasonalized = (
                df.drop(["time", "site"], axis=1).groupby(by=[df.year, df.site]).transform(lambda x: x - x.mean())
            )
            df_deseasonalized = df_deseasonalized.groupby(by=[df.dayofyear, df.site]).transform(lambda x: x - x.mean())

        arr_all_data = np.zeros((df.shape[0], len(list_control) + len(list_variables)))
        for k, var in enumerate(list_control):
            df_deseasonalized[var] -= df_deseasonalized[var].mean()
            df_deseasonalized[var] /= df_deseasonalized[var].std()
            arr_all_data[:, k] = df_deseasonalized[var]
        for k, var in enumerate(list_variables):
            df_deseasonalized[var] -= df_deseasonalized[var].mean()
            df_deseasonalized[var] /= df_deseasonalized[var].std()
            arr_all_data[:, len(list_control) + k] = df_deseasonalized[var]
        arr_all_data = arr_all_data.astype(np.float32)

        self._data_train, self._data_test = train_test_split(arr_all_data, test_size=self.val_split, random_state=1)
        self._data_train, self._data_val = train_test_split(
            self._data_train, test_size=self.val_split / (1 - self.val_split), random_state=1
        )

    def train_dataloader(self, accelerator):

        return DataLoader(
            dataset=self._data_train,
            batch_size=self.batch_size,
            shuffle=True,
            generator=torch.Generator(device=accelerator.device),
            drop_last=True,
            **self._shared_dataloader_kwargs(),
        )

    def val_dataloader(self, accelerator):

        return DataLoader(
            dataset=self._data_val,
            batch_size=self.batch_size,
            shuffle=False,
            generator=torch.Generator(device=accelerator.device),
            drop_last=True,
            **self._shared_dataloader_kwargs(),
        )

    def test_dataloader(self, accelerator):

        return DataLoader(
            dataset=self._data_test,
            batch_size=self.batch_size,
            shuffle=False,
            generator=torch.Generator(device=accelerator.device),
            drop_last=True,
            **self._shared_dataloader_kwargs(),
        )
