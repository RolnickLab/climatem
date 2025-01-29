import json
from climatem import TUNING_CONFIGS

# configs that should be changed for tuning:
tuning_grid = {
    "d_z": [30, 70, 110, 150],
    "tau": [3, 7, 9],
    "nonlinear_mixing": [True, False],
}

if __name__ == "__main__":
    print(tuning_grid)
    # load default configs
    with open(TUNING_CONFIGS / "default_configs.json") as f:
        default_configs = json.load(f)
    
    # iterate over grid 
    for d_z in tuning_grid["d_z"]:
        for tau in tuning_grid["tau"]:
            for nonlinear_mixing in tuning_grid["nonlinear_mixing"]:
                # create new json name 
                filename = f"jk_dz-{d_z}_tau-{tau}_nonlinearmixing-{nonlinear_mixing}.json"

                # update configs
                curr_configs = default_configs
                curr_configs["d_z"] = d_z 
                curr_configs["tau"] = tau 
                curr_configs["nonlinear_mixing"] = nonlinear_mixing

                # store as json file 
                with open(TUNING_CONFIGS / filename, 'w') as f:
                    json.dump(curr_configs, f)

    

    


