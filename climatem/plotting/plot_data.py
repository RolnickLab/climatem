import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cv2
import matplotlib.pyplot as plt
import numpy as np

# NOTE: SH has changed this to plot a historical series of data from 1850-2014
# and to save as mp4
def plot_species(data, coordinates, var, out_dir, num_video):
    """Plot the given species data on a map with a colorbar."""

    print("plotting")
    # min_temp = np.floor(100 * data.min()) / 100
    # max_temp = np.ceil(100 * data.max()) / 100

    # can't use ceiling for precipitation due to units...
    if var == "tas":
        min_temp = np.floor(data.min())
        max_temp = np.ceil(data.max())
    elif var == "pr":
        min_temp = data.min()
        max_temp = data.max()
    else:
        print("Variable not recognized, please choose 'tas' or 'pr'")

    longitudes, latitudes = np.unique(coordinates[:, 1]), np.unique(coordinates[:, 0])

    for y in range(65):
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

            # NOTE: I should probably set the cmap here to be better, and maybe to depend on the variable
            # Add filled contours of the emissions data to the map (extend='both') is an option to extend colorbar in both directions
            # c = ax.contourf(longitudes, latitudes, data_ym, transform=ccrs.PlateCarree(), cmap='RdBu_r', vmin=min_temp, vmax=max_temp)
            c = ax.contourf(longitudes, latitudes, data_ym, transform=ccrs.PlateCarree(), cmap="RdBu_r", levels=levels)

            # do I really want this to be a contourf plot?

            # Add a colorbar to the map
            cbar = fig.colorbar(c, ax=ax, orientation="vertical", fraction=0.04, pad=0.05)

            if var == "tas":
                cbar.set_label("Temperature", rotation=270, labelpad=15)
            elif var == "pr":
                cbar.set_label("Precipitation", rotation=270, labelpad=15)
            else:
                # Need to make this better.
                cbar.set_label("Variable?", rotation=270, labelpad=15)

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

    for y in range(65):
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

