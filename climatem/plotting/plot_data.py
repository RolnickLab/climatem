import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cv2
import matplotlib.pyplot as plt
import numpy as np
from tqdm import trange


# NOTE: SH has changed this to plot a historical series of data from 1850-2014
# and to save as mp4
def plot_species(
    data, coordinates, list_var, out_dir, name_video, year_range=100, start_year=1850, num_levels=16, cmap="RdBu_r"
):
    """Plot the given species data on a map with a colorbar."""

    print("plotting")
    # min_temp = np.floor(100 * data.min()) / 100
    # max_temp = np.ceil(100 * data.max()) / 100

    # can't use ceiling for precipitation due to units...
    # if var == "tas":
    #     min_temp = np.floor(data.min())
    #     max_temp = np.ceil(data.max())
    # elif var == "pr":
    min_temp = np.nanmin(data, axis=(0, 1, 3, 4))
    max_temp = np.nanmax(data, axis=(0, 1, 3, 4))

    levels_allvar = []
    for mint, maxt in zip(min_temp, max_temp):
        levels_allvar.append(np.linspace(mint, maxt, num_levels + 1))

    longitudes, latitudes = np.unique(coordinates[:, 1]), np.unique(coordinates[:, 0])

    # if longitudes.max() > 180:
    #     longitudes[longitudes > 180] -= 360
    # shape = (len(latitudes), len(longitudes))

    longitudes = np.concatenate([longitudes, longitudes[:1] + 360])
    data = np.concatenate([data, data[:, :, :, :, :1]], axis=-1)

    nrows = 1 if len(list_var) == 1 else int(np.ceil(len(list_var) / 2))
    ncols = 1 if len(list_var) == 1 else 2

    var_name = str(list_var)[1:-1].translate({ord("'"): None}).translate({ord(","): None}).translate({ord(" "): None})

    for y in trange(year_range):
        year = start_year + y
        for m in range(12):
            # if num_video == "after_causal":
            #     data_ym = data[y * 12 + m].reshape(shape)
            # else:
            # data_ym = data[y, m].reshape(shape)

            # Create a figure with specified size and axis with a map projection
            fig, axs = plt.subplots(nrows, ncols, subplot_kw={"projection": ccrs.PlateCarree()}, figsize=(12, 8))
            axs = axs.flatten()

            if len(axs) > len(list_var):
                list_var.append(None)

            for i, (var, levels, ax) in enumerate(zip(list_var, levels_allvar, axs)):
                data_ym = data[y, m, i]
                if var:
                    ax.coastlines()
                    # NOTE: I should probably set the cmap here to be better, and maybe to depend on the variable
                    # Add filled contours of the emissions data to the map (extend='both') is an option to extend colorbar in both directions
                    # c = ax.contourf(longitudes, latitudes, data_ym, transform=ccrs.PlateCarree(), cmap='RdBu_r', vmin=min_temp, vmax=max_temp)
                    c = ax.contourf(
                        longitudes,
                        latitudes,
                        data_ym,
                        transform=ccrs.PlateCarree(),
                        cmap=cmap,
                        levels=levels,
                        antialiased=True,
                    )

                    # do I really want this to be a contourf plot?

                    # Add a colorbar to the map
                    fig.colorbar(c, ax=ax, orientation="vertical", fraction=0.04, pad=0.05)  # cbar =

                    # if var == "tas":
                    #     cbar.set_label("Temperature", rotation=270, labelpad=15)
                    # elif var == "pr":
                    #     cbar.set_label("Precipitation", rotation=270, labelpad=15)
                    # else:
                    #     # Need to make this better.
                    #     cbar.set_label("Variable?", rotation=270, labelpad=15)
                    ax.set_title(var)

                    # Add some map features for context
                    ax.add_feature(cfeature.BORDERS, linestyle=":")
                    ax.add_feature(cfeature.COASTLINE)
                    ax.add_feature(cfeature.LAND, edgecolor="black")

                    # Label the axes with latitude and longitude values
                    ax.set_xticks(np.arange(-180, 181, 30), crs=ccrs.PlateCarree())
                    ax.set_yticks(np.arange(-90, 91, 30), crs=ccrs.PlateCarree())
                    ax.set_xticklabels(np.arange(-180, 181, 30))
                    ax.set_yticklabels(np.arange(-90, 91, 30))
                    # ax.gridlines(draw_labels=False)

            # Set the title
            # ax.set_title(f"{var} averaged over {years}")

            plt.suptitle(f"Year {year}, month {m}")

            fname = f"{out_dir}/{var_name}_{year}_{m}_plot.png"

            plt.savefig(fname)
            plt.close()

    img_array = []

    for y in trange(year_range):
        year = start_year + y
        for m in range(12):
            filename = f"{out_dir}/{var_name}_{year}_{m}_plot.png"
            # filename = f"{out_dir}/plot_{i}.png"
            img = cv2.imread(filename)
            height, width, layers = img.shape
            size = (width, height)
            img_array.append(img)

    out = cv2.VideoWriter(f"{out_dir}/{name_video}.mp4", cv2.VideoWriter_fourcc(*"DIVX"), 15, size)

    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()


# NOTE: SH has added this function to plot the anomaly of the data compared to some baseline, often the monthly mean
def plot_species_anomaly(data, coordinates, var, out_dir, num_video, method="monthly_mean"):
    """
    Plot the given species anomaly data on a map with a colorbar.

    The anomaly is the monthly anomaly from the monthly mean for the whole time period of the data.
    """

    print("plotting")

    longitudes, latitudes = np.unique(coordinates[:, 1]), np.unique(coordinates[:, 0])

    # compute the monthly anomaly here, but simply subtracting the monthly mean
    if method == "monthly_mean":
        data = data - np.mean(data, axis=0)

    # also possible to compute the monthly anomaly scaled by the standard dev. of the data for that month
    elif method == "monthly_scaled":
        data = (data - np.mean(data, axis=0)) / np.std(data, axis=0)

    else:
        print("Method not recognized, please choose 'monthly_mean' or 'monthly_scaled'")
    # also compute the monthly anomaly adjusted for the effect of emissions
    # detrended_data = data - co2_sum * coefficient

    # possible option for min and max values, they are simply for the colorbar
    # min_temp = np.floor(100 * data.min()) / 100
    # max_temp = np.ceil(100 * data.max()) / 100
    min_temp = np.floor(data.min())
    max_temp = np.ceil(data.max())

    for y in range(100):
        year = 1850 + y
        print(year)
        for m in range(12):
            if num_video == "after_causal":
                data_ym = data[y * 12 + m].reshape((96, 144))
            else:
                data_ym = data[y, m].reshape((96, 144))
            # Create a figure with specified size and axis with a map projection
            fig, ax = plt.subplots(subplot_kw={"projection": ccrs.PlateCarree()}, figsize=(12, 8))
            ax.coastlines()

            vmin, vmax = min_temp, max_temp  # Minimum and maximum values for colorbar

            # Define number of color levels (adjust based on your colormap)
            num_levels = 16
            levels = np.linspace(vmin, vmax, num_levels + 1)

            # Add filled contours of the emissions data to the map
            # c = ax.contourf(longitudes, latitudes, data_ym, transform=ccrs.PlateCarree(), cmap='RdBu_r', vmin=min_temp, vmax=max_temp)
            c = ax.contourf(longitudes, latitudes, data_ym, transform=ccrs.PlateCarree(), cmap="RdBu_r", levels=levels)

            # do I really want this to be a contourf plot?

            # Add a colorbar to the map
            cbar = fig.colorbar(c, ax=ax, orientation="vertical", fraction=0.04, pad=0.05)
            cbar.set_label("Temperature", rotation=270, labelpad=15)

            # Add some map features for context
            ax.add_feature(cfeature.BORDERS, linestyle=":")
            ax.add_feature(cfeature.COASTLINE)
            ax.add_feature(cfeature.LAND, edgecolor="black")

            # Label the axes with latitude and longitude values
            ax.set_xticks(np.arange(-180, 181, 30), crs=ccrs.PlateCarree())
            ax.set_yticks(np.arange(-90, 91, 30), crs=ccrs.PlateCarree())
            ax.set_xticklabels(np.arange(-180, 181, 30))
            ax.set_yticklabels(np.arange(-90, 91, 30))
            ax.gridlines(draw_labels=False)

            # Set the title
            # ax.set_title(f"{var} averaged over {years}")

            fname = f"{out_dir}/{var}_{year}_{m}_plot.png"
            plt.title(f"YEAR : {year}, MONTH: {m}")
            plt.savefig(fname)
            plt.close()

    img_array = []

    for y in range(100):
        year = 1850 + y
        print(year)
        # for j, month in enumerate(months_list):
        for m in range(12):
            filename = f"{out_dir}/{var}_{year}_{m}_plot.png"
            # filename = f"{out_dir}/plot_{i}.png"
            img = cv2.imread(filename)
            height, width, layers = img.shape
            size = (width, height)
            img_array.append(img)

    out = cv2.VideoWriter(f"{out_dir}/video_{num_video}.mp4", cv2.VideoWriter_fourcc(*"DIVX"), 15, size)

    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()
