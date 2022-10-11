import numpy as np
import open3d
import argparse
SRCPATH = "flanges"



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-f", "--file", type = str, required = True, help = "PCD file name")
    args = vars(ap.parse_args())
    
    cloud = open3d.read_point_cloud(os.path.join(SRCPATH, args["file"])) # Read the point cloud
    open3d.draw_geometries([cloud]) # Visualize the point cloud     

if __name__ == "__main__":
    main()
