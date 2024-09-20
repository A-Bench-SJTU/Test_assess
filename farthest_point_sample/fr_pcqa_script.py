import open3d as o3d
import numpy as np
import os
import time
import random
import pandas as pd
from p2plane import p2plane
from p2point import p2point,p2point_hausdorf
from psnr_y import psnr_y

def get_score(database):
    if database == 'sjtu':
        dis_path = '/home/zzc/3dqa/pointnet2/database/sjtu/'
        ref_path = '/home/zzc/3dqa/reference/sjtu/'
        info_path = '/home/zzc/3dqa/pointnet2/csvfiles/sjtu_data_info'
        ex_id = 9
    elif database == 'wpc':
        dis_path = '/home/zzc/3dqa/pointnet2/database/wpc/'
        ref_path = '/home/zzc/3dqa/reference/wpc/'
        info_path = '/home/zzc/3dqa/pointnet2/csvfiles/wpc_data_info_with_prefix'
        ex_id = 5

    

    #for i in range(ex_id):
    for i in range(4,ex_id):
        info = pd.read_csv(os.path.join(info_path,'test_'+str(i+1)+'.csv'))
        names = info['name']
        scores = []
        p2point_ = []
        p2point_hausdorf_ = []
        p2plane_ = []
        psnr_y_ = []
        for name in names:
            if database == 'wpc':
                ref_name = os.path.join(ref_path,name.split('/')[0]+'.ply')
                dis_name = os.path.join(dis_path,name)
                print(dis_name)
            elif database == 'sjtu':
                ref_name = os.path.join(ref_path,name.split('_')[0]+'.ply')
                dis_name = os.path.join(dis_path,name.split('_')[0],name)
                print(dis_name)

            start = time.time()
            p2point_.append(p2point(ref_name,dis_name))
            p2point_hausdorf_.append(p2point_hausdorf(ref_name,dis_name))
            p2plane_.append(p2plane(ref_name,dis_name))
            psnr_y_.append(psnr_y(ref_name,dis_name))
            end = time.time()
            print('Consuming seconds /s :' + str(end-start))
        
        result = pd.DataFrame({'name':names,'p2point':p2point_,'p2point_hausdorf':p2point_hausdorf_,'p2plane':p2plane_,'psnr_y':psnr_y_})
        print(result)
        result.to_csv(database + '/' + 'test_' + str(i+1) + '_fr_score.csv',index = None)    


get_score('sjtu')
#get_score('wpc')
