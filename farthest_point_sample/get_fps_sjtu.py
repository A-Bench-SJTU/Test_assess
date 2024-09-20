import open3d as o3d
import numpy as np
import os
import time
import random
def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc


def farthest_point_sample(point, npoint):
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    N, D = point.shape
    xyz = point[:,:3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point = point[centroids.astype(np.int32)]
    return point



def get_farthest_point_sample(pcd_name,npy_name):
    pcd = o3d.io.read_point_cloud(pcd_name)
    point_set = np.asarray(pcd.points)
    #print(point_set)
    point_set = farthest_point_sample(point_set, 4096)
    point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])
    #print(point_set)
    np.save(npy_name,point_set)

path = '/home/zzc/3dqa/pointnet2/database/sjtu'
out_path = '/home/zzc/3dqa/pointnet2/database/sjtu_fps_4096'
files = os.listdir(path)
for file in files[4:]:
    file_path = os.path.join(path,file)
    objs = os.listdir(file_path)
    for obj in objs:
        start = time.time()
        pcd_name = os.path.join(file_path,obj)
        npy_name = os.path.join(out_path,obj.split('.')[0] + '.npy')
        print(pcd_name)
        get_farthest_point_sample(pcd_name,npy_name)
        end = time.time()
        print('Consuming seconds /s :' + str(end-start))