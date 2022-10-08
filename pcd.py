import numpy as np
import open3d
import argparse
srcpath = "flanges"

ap = argparse.ArgumentParser()
ap.add_argument("-f", "--file", type = str, required = True, help = "PCD file name")
args = vars(ap.parse_args())
srcpath = os.path.join(srcpath, args["file"])

def main():
    
    cloud = open3d.read_point_cloud(srcpath) # Read the point cloud
    open3d.draw_geometries([cloud]) # Visualize the point cloud     

if __name__ == "__main__":
    main()
