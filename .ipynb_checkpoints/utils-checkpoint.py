import laspy
import open3d as o3d
import numpy as np
import os
from pathlib import Path

# Leer archivo LAS


def las_to_ply(filename:str):
    file_folder = os.path.dirname(filename)
    name = Path(filename).stem
    print(file_folder)
    las = laspy.read(filename)
    points = np.vstack((las.x, las.y, las.z)).transpose()

    # Crear nube de puntos en Open3D
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # Guardar en formato PLY
    o3d.io.write_point_cloud(file_folder + '/' + name +".ply", pcd)