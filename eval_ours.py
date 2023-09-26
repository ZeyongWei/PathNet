import scipy.spatial as sp
import numpy as np
import torch

import os

from Customer_Module.chamfer_distance.dist_chamfer import chamferDist
from plyfile import PlyData, PlyElement
nnd = chamferDist()

import logging

def log_string(str):
    logger.info(str)
    print(str)

name_dirs = ['benchmark81_10000','benchmark81_20000','benchmark81_50000','kinect_fusion','kinect_v1','kinect_v2']

name_dir = 'benchmark81_20000'

model = ''

iter_num = 2

results_dir = '//' + model + '/' + name_dir + '/'

back_logs = ['_output_end','_output_end_output_end']

back_log = back_logs[iter_num-1]

if(not os.path.exists('./data/test_data/' + name_dir + '/eval/' + model +'/')):
   os.makedirs('./data/test_data/' + name_dir + '/eval/' + model +'/')

if(os.path.exists('./data/test_data/' + name_dir + '/eval/' + model +'/eval_log_' + name_dir +'_' + str(iter_num) + '.csv')):
    os.remove('./data/test_data/' + name_dir + '/eval/' + model +'/eval_log_' + name_dir +'_' + str(iter_num) + '.csv')

logger = logging.getLogger("Eval"+'_' + model + '_' + name_dir + '_' + str(iter_num))
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler = logging.FileHandler('./data/test_data/' + name_dir + '/eval/' + model +'/eval_log_' + name_dir +'_' + str(iter_num) + '.csv')
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)


def Eval_With_Charmfer_Distance():
    log_string('************Errors under Chamfer Distance************')
    for shape_id, shape_name in enumerate(shape_names):
        if(name_dir.split('_')[0]=='benchmark81'):
            gt_pts = np.loadtxt(os.path.join('./data/test_data/gt/'+ name_dir +'/', shape_name.split('_0.')[0] + '.xyz'))
        else:
            gt_pts = np.loadtxt(os.path.join('./data/test_data/gt/'+ name_dir +'/', shape_name.split('_noisy')[0] + '.xyz'))
        pred_pts = np.loadtxt(os.path.join('./data/results/'+ results_dir +'/', shape_name + back_log + '.xyz'))[:,:3]
        with torch.no_grad():
            gt_pts_cuda = torch.from_numpy(np.expand_dims(gt_pts, axis=0)).cuda().float()
            pred_pts_cuda = torch.from_numpy(np.expand_dims(pred_pts, axis=0)).cuda().float()
            dist1, dist2 = nnd(pred_pts_cuda, gt_pts_cuda)
            chamfer_errors = torch.mean(dist1, dim=1) + torch.mean(dist2, dim=1)

            log_string('%12s  %.3f' % (shape_names[shape_id], round(chamfer_errors.item() * 100000, 3)))

def Eval_With_Mean_Square_Error():
    log_string('************Errors under Mean Square Error************')
    for shape_id, shape_name in enumerate(shape_names):
        if(name_dir.split('_')[0]=='benchmark81'):
            gt_pts = np.loadtxt(os.path.join('./data/test_data/gt/' + name_dir +'/', shape_name.split('_0.')[0] + '.xyz'))
        else:
            gt_pts = np.loadtxt(os.path.join('./data/test_data/gt/' + name_dir +'/', shape_name.split('_noisy')[0] + '.xyz'))
        gt_pts_tree = sp.cKDTree(gt_pts)
        pred_pts = np.loadtxt(os.path.join('./data/results/'+ results_dir +'/', shape_name + back_log +'.xyz'))[:,:3]
        pred_dist, _ = gt_pts_tree.query(pred_pts, 10)

        log_string('%12s  %.3f' % (shape_names[shape_id], round(pred_dist.mean() * 1000, 3)))


if __name__ == '__main__':

    with open(os.path.join('./data/test_data/gt/'+ name_dir +'/', 'test.txt'), 'r') as f:
        shape_names = f.readlines()
    shape_names = [x.strip() for x in shape_names]
    shape_names = list(filter(None, shape_names))

    Eval_With_Charmfer_Distance()
    Eval_With_Mean_Square_Error()

