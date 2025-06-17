from climatem.data_loader.causal_datamodule import CausalClimateDataModule
from pathlib import Path

# get root path
root_path = Path(__file__).parent.parent
print(root_path)

#TODO look at dimensionality / shape of the data
# write loop to go over each item of the dataset as if i am the model
# get random sample from dataset and inspect x and y 
# make sure that get causal data is actually using sequential inputs
# plot the data
# check that the normalization works
# compare size of test set to train set
# check where it gest loaded in main_picabu as dataloader - look at vars
# TODO global normalization

dl = CausalClimateDataModule(
    output_save_dir=f"{root_path}/../scratch/data/SAVAR_DATA_TEST",
    in_var_ids=["savar"],
    train_models="savar",
    train_years="2015-2100",
    train_historical_years="1950-2014",
    lat=20,
    lon=20,
    tau=5,
    # global_normalization=True,
    # seasonality_removal=True,
    reload_climate_set_data=True,
    time_len=10_000,
    comp_size=10,
    noise_val=0.2,
    n_per_col=2,
    difficulty="easy",
    seasonality=False,
)

dl.setup()

breakpoint()

