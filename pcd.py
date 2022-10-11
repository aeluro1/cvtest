import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import proj3d
import argparse
import os


SRCPATH = "flanges"

def main():
    # Get .xyz file name
    ap = argparse.ArgumentParser()
    ap.add_argument("-f", "--file", type = str, required = True, help = "PCD file name")
    args = vars(ap.parse_args())

    # Extract points into Nx3 array
    data = np.genfromtxt(os.path.join(SRCPATH, args["file"]), delimiter = " ")
    x = data[:, 0]
    y = data[:, 1]
    z = data[:, 2]

    # Plot
    plt.rcParams.update({"figure.autolayout": True})
    
    fig = plt.figure(figsize = (10, 10))
    ax = fig.add_subplot(111, projection = "3d")

    ax.scatter(x, y, z)
    xticks = ax.get_xticks()
    ax.set_zticks(xticks)
    #fig.colorbar(ax)
    plt.show()
##    cloud = open3d.read_point_cloud(os.path.join(SRCPATH, args["file"])) # Read the point cloud
##    open3d.draw_geometries([cloud]) # Visualize the point cloud

if __name__ == "__main__":
    main()
