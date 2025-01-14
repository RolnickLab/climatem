# Here we try to modify the climate_data_loader so that we can use data from multiple ensemble members of a climate model, and indeed across climate models.

import json
import numpy as np
from typing import Optional


import torch

from climatem.climate_datamodule_explore_ensembles_multigpu import ClimateDataModule

# Here we replace the original import with a new experimental import.
#from emulator.src.data.climate_dataset import Input4MipsDataset, CMIP6Dataset
from climatem.climate_dataset_explore_ensembles import CMIP6Dataset, Input4MipsDataset

from climatem.constants import AVAILABLE_MODELS_FIRETYPE, OPENBURNING_MODEL_MAPPING


class CausalDataset(torch.utils.data.Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __getitem__(self, index: int):
        return self.x[index], self.y[index]
    
    def __len__(self):
        return len(self.x)


class CausalClimateDataModule(ClimateDataModule):
    """
    This class inherits from the ClimateDataModule class and uses the same
    initialization parameters. The setup method is overwritten and performs data
    preprocessing for causal discovery models.
    """

    def __init__(self, tau=5, num_months_aggregated=1, train_val_interval_length=100, **kwargs):
        super().__init__(self)

        self.hparams.test_models = (
            None if self.hparams.test_models else self.hparams.train_models
        )

        self.tau = tau
        self.num_months_aggregated = num_months_aggregated
        self.train_val_interval_length = train_val_interval_length
        self.shuffle_train = False  # need to keep order for causal train / val splits


    @staticmethod
    def years_to_list(years_str):
        """
        Convert years input to list of years.
        """
        if years_str is None:
            return []
        elif isinstance(years_str, int):
            return [years_str]
        elif isinstance(years_str, str):
            print(years_str)
            if len(years_str) != 9:
                raise ValueError(
                    "Years string must be in the format xxxx-yyyy (eg. 2015-2100)."
                )

            years = years_str.split("-")
            min_year, max_year = int(years[0]), int(years[1])
            return np.arange(min_year, max_year + 1)
        else:
            raise ValueError(
                f"years_str must be int, str, or None, not {type(years_str)}"
            )



    def setup(self, stage: Optional[str] = None):
        if stage in ["fit", "validate", None]:
            openburning_specs = (
                OPENBURNING_MODEL_MAPPING[self.hparams.train_models]
                if self.hparams.train_models in AVAILABLE_MODELS_FIRETYPE
                else OPENBURNING_MODEL_MAPPING["other"]
            )

            train_years = self.years_to_list(self.hparams.train_years)
            train_historical_years = self.years_to_list(
                self.hparams.train_historical_years
            )

            #NOTE:(seb) 23rd May, changing to include psl too... this is limiting, 
            # and tells us just to do this if we are looking at tas or pr, 
            # we basically just ignore the input4mips dataset
            if "tas" in self.hparams.in_var_ids or "pr" in self.hparams.in_var_ids or "psl" in self.hparams.in_var_ids or "ts" in self.hparams.in_var_ids:
                print(self.hparams.train_scenarios)
                train_val_input4mips = CMIP6Dataset(
                    years=train_years,
                    coordinates_path = self.hparams.icosahedral_coordinates_path,
                    historical_years=train_historical_years,
                    data_dir=self.hparams.data_dir,
                    climate_model=self.hparams.train_models,
                    #num_ensembles=1, #TODO: extend to more than one - NOTE:(seb) this shall be done! Ok going to do this now, 20th August 2024.
                    num_ensembles=self.hparams.num_ensembles,
                    variables=self.hparams.in_var_ids,
                    scenarios=self.hparams.train_scenarios,
                    channels_last=self.hparams.channels_last,
                    openburning_specs=openburning_specs,
                    mode="train+val",
                    output_save_dir=self.hparams.output_save_dir,
                )
            else:
                # NOTE:(seb) if I am here, I have probably make an error and I do not have the right input variable above directing us to the CMIP6Dataset
                train_val_input4mips = Input4MipsDataset(
                    years=train_years,
                    coordinates_path = self.hparams.icosahedral_coordinates_path,
                    historical_years=train_historical_years,
                    data_dir=self.hparams.data_dir,
                    variables=self.hparams.in_var_ids,
                    scenarios=self.hparams.train_scenarios,
                    channels_last=self.hparams.channels_last,
                    openburning_specs=openburning_specs,
                    mode="train+val",
                    output_save_dir=self.hparams.output_save_dir,
                )

            ratio_train = 1 - self.hparams.val_split

            # NOTE:(seb) - this may break if we are using historical data, I guess?
            # NOTE:(seb) adding num ensembles here for the dataloading
            # NOTE:(seb) num_years is overwritten in the get_causal_data function
            train, val = train_val_input4mips.get_causal_data(
                tau=self.tau,
                channels_last=self.hparams.channels_last,
                num_vars=len(self.hparams.in_var_ids),
                num_scenarios=len(self.hparams.train_scenarios),
                num_ensembles=self.hparams.num_ensembles,
                num_years=len(train_years),
                ratio_train=ratio_train,
                num_months_aggregated=self.num_months_aggregated,
                interval_length=self.train_val_interval_length,
                mode="train+val",
            )


            train_x, train_y = train
            train_x = train_x.reshape((train_x.shape[0], train_x.shape[1], train_x.shape[2], -1))
            train_y = train_y.reshape((train_y.shape[0], train_y.shape[1], train_y.shape[2], -1))

            self.d = train_x.shape[2]
            self._data_train = CausalDataset(train_x, train_y)
            self.n_train = train_x.shape[0]

            if val is not None:
                val_x, val_y = val
                val_x = val_x.reshape((val_x.shape[0], val_x.shape[1], val_x.shape[2], -1))
                val_y = val_y.reshape((val_y.shape[0], val_y.shape[1], val_y.shape[2], -1))
                self._data_val = CausalDataset(val_x, val_y)

            self.coordinates = train_val_input4mips.coordinates


        if stage in ["test", None]:
            openburning_specs = {
                test_model: (
                    OPENBURNING_MODEL_MAPPING[test_model]
                    if test_model in AVAILABLE_MODELS_FIRETYPE
                    else OPENBURNING_MODEL_MAPPING["other"]
                )
                for test_model in self.hparams.test_models
            }

            test_years = self.years_to_list(self.hparams.test_years)
            test_historical_years = self.years_to_list(None)

            self._data_test = []
            for test_scenario in self.hparams.test_scenarios:
                for test_model in self.hparams.test_models:
                    # NOTE:(seb) adding psl here...note that this should probably be extended to include other variables...************
                    # NOTE:(seb) THIS JUST LOOKS AT train_years and train_historical_years - test_years is totally ignored for this dataset!!!
                    if "tas" in self.hparams.in_var_ids or "pr" in self.hparams.in_var_ids or "psl" in self.hparams.in_var_ids or "ts" in self.hparams.in_var_ids:
                        test_input4mips = CMIP6Dataset(
                            years=train_years,
                            historical_years=train_historical_years,
                            coordinates_path = self.hparams.icosahedral_coordinates_path,
                            data_dir=self.hparams.data_dir,
                            climate_model=self.hparams.train_models,
                            # NOTE:(seb) this has now been added
                            num_ensembles=self.hparams.num_ensembles,  # TODO: extend to more than one NOTE:(seb) this shall be done!
                            variables=self.hparams.in_var_ids,
                            scenarios=self.hparams.train_scenarios,
                            channels_last=self.hparams.channels_last,
                            openburning_specs=openburning_specs,
                            mode="train+val",
                            output_save_dir=self.hparams.output_save_dir,
                            seasonality_removal=self.hparams.seasonality_removal
                        )
                    else:
                        test_input4mips = Input4MipsDataset(
                            years=test_years,
                            coordinates_path = self.hparams.icosahedral_coordinates_path,
                            historical_years=test_historical_years,
                            data_dir=self.hparams.data_dir,
                            variables=self.hparams.in_var_ids,
                            scenarios=[test_scenario],
                            channels_last=self.hparams.channels_last,
                            openburning_specs=openburning_specs[test_model],
                            mode="test",
                            output_save_dir=self.hparams.output_save_dir,
                        )

                    test = test_input4mips.get_causal_data(
                        tau=self.tau,
                        channels_last=self.hparams.channels_last,
                        num_vars=len(self.hparams.in_var_ids),
                        num_scenarios=1,
                        num_ensembles=self.hparams.num_ensembles,
                        num_years=len(train_years),
                        num_months_aggregated=self.num_months_aggregated,
                        mode="test",
                    )

                    test_x, test_y = test
                    test_x = test_x.reshape((test_x.shape[0], test_x.shape[1], test_x.shape[2], -1))
                    test_y = test_y.reshape((test_y.shape[0], test_y.shape[1], test_y.shape[2], -1))
                    test_scenario_data = CausalDataset(test_x, test_y)
                    self._data_test.append(test_scenario_data)
                    # self.train_years = train_val_input4mips.
                    self.coordinates = test_input4mips.coordinates

        if stage in ["predict", None]:
            self._data_predict = self._data_test


# TODO: CHANGE BELOW --> 2 json files, main() function although this is not a script.... We need one json and one class that we call in the scripts
# Also there's too many directories, a bit of a mess
# + should add some assert / os path.exists / makedirs to check that directories exist

def main():
    # Has to be json file
    # I guess this is setting a default...
    # TODO: REMOVE THIS!! OR MAKE IT IN THE SAME REPO / Better to have only 1 json files + all the data stored in the same place
    config_fname = "/home/mila/j/julien.boussard/causal_model/causalpaca/emulator/configs/datamodule/climate_1850_2015_tas.json"
    print(config_fname)
    with open(config_fname) as f:
        data_params = json.load(f)
        
    datamodule = CausalClimateDataModule(**data_params)
    datamodule.setup()

    train_dataloader = iter(datamodule.train_dataloader())
    x, y = next(train_dataloader)
    # SHAPES:
    # x:
    print(x.shape, y.shape)

    dataset = datamodule._data_train
    print(len(dataset))

    # TODO: validate on entire set [set batch size to len(dataset)] or batch in valid_step?
    # val_dataloader = datamodule.val_dataloader()
    # test_dataloader = datamodule.test_dataloader()


if __name__ == "__main__":
    main()