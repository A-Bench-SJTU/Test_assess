import open3d as o3d
import numpy as np
import os
import time
import random
def visualize(points):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.paint_uniform_color([0.5, 0.5, 0.5])
    o3d.visualization.draw_geometries([pcd])

def pc_normalize(pc):
    l = pc.shape[0]
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc

def farthest_point_sample(point, patch_size):
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    N, D = point.shape
    npoint = int(N/patch_size) + 1
    if N < npoint:
        idxes = np.hstack((np.tile(np.arange(N), npoint//N), np.random.randint(N, size=npoint%N)))
        return point[idxes, :]

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

def knn_patch(pcd_name, patch_size = 2048):
    pcd = o3d.io.read_point_cloud(pcd_name)
    # nomalize pc and set up kdtree
    points = pc_normalize(np.array(pcd.points))
    pcd.points = o3d.utility.Vector3dVector(points)
    kdtree = o3d.geometry.KDTreeFlann(pcd)

    fps_point = farthest_point_sample(points,patch_size)
    
    
    point_size = fps_point.shape[0]


    patch_list = []

    for i in  range(point_size):
        [_,idx,dis] = kdtree.search_knn_vector_3d(fps_point[i],patch_size)
        #print(pc_normalize(np.asarray(point)[idx[1:], :]))
        patch_list.append(np.asarray(points)[idx[:], :]) 
    
    #visualize(all_point(np.array(patch_list)))
    #visualize(point)
    return np.array(patch_list)


# path = '/home/zzc/3dqa/pointnet2/database/wpc'
# out_path = '/home/zzc/3dqa/pointnet2/database/wpc_patch_2048'
# files = os.listdir(path)
# for file in files[:]:
#     file_path = os.path.join(path,file)
#     objs = os.listdir(file_path)
#     for obj in objs:
#         start = time.time()
#         pcd_name = os.path.join(file_path,obj)
#         npy_name = os.path.join(out_path,obj.split('.')[0] + '.npy')
#         print(pcd_name)
#         patch = knn_patch(pcd_name)
#         np.save(npy_name,patch)
#         end = time.time()
#         print('Consuming seconds /s :' + str(end-start))

path = '/home/zzc/vqa_pc/database/lspcqa/samples_with_MOS'
out_path = '/home/zzc/3dqa/pointnet2/database/lspcqa_patch_2048'
files = os.listdir(path)
for obj in files:
    start = time.time()
    pcd_name = os.path.join(path,obj)
    npy_name = os.path.join(out_path,obj.split('.')[0] + '.npy')
    print(pcd_name)
    patch = knn_patch(pcd_name)
    np.save(npy_name,patch)
    end = time.time()
    print('Consuming seconds /s :' + str(end-start))