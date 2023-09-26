"""
Author: Wei
Date: Mar 2022
"""
import argparse
import os
from data_utils.DataLoader import PatchDataset
import torch
import datetime
import logging
from pathlib import Path
import sys
import importlib
import shutil
from tqdm import tqdm
import provider
import numpy as np
from scipy import spatial
from glob import glob
from data_utils.Pointfilter_Utils import pca_alignment
import time
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))


def parse_args():
    parser = argparse.ArgumentParser('Model')
    parser.add_argument('--model', type=str, default='model', help='model name [default: model]')
    parser.add_argument('--batch_size', type=int, default=100, help='Batch Size during training [default: 16]')
    parser.add_argument('--epoch',  default=1000, type=int, help='Epoch to run [default: 251]')
    parser.add_argument('--learning_rate', default=0.000001, type=float, help='Initial learning rate [default: 0.001]')
    parser.add_argument('--gpu', type=str, default='0', help='GPU to use [default: GPU 0]')
    parser.add_argument('--optimizer', type=str, default='Adam', help='Adam or SGD [default: Adam]')
    parser.add_argument('--optimizer2', type=str, default='Adam', help='Adam or SGD [default: Adam]')
    parser.add_argument('--log_dir', type=str, default='model', help='Log path [default: None]')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='weight decay [default: 1e-4]')
    parser.add_argument('--npoint', type=int,  default=128, help='Point Number [default: 2048]')
    parser.add_argument('--normal', action='store_true', default=False, help='Whether to use normal information [default: False]')
    parser.add_argument('--step_size', type=int,  default=20, help='Decay step for lr decay [default: every 20 epochs]')
    parser.add_argument('--lr_decay', type=float,  default=0.5, help='Decay rate for lr decay [default: 0.5]')
    parser.add_argument('--use_random_path', type=int, default=0, help='wether use random path 0 no ,1 yes, 2 all 0, 3 all 1 ')
    parser.add_argument('--block_num', type=int, default=6, help='num of denosier block')
    parser.add_argument('--path_num', type=int, default=2, help='path num of each denosier block')

    return parser.parse_args()

def pca_normalize(pc):
    normalize_data = np.zeros(pc.shape, dtype=np.float32)
    martix_inv = np.zeros((pc.shape[0],3,3),dtype=np.float32)

    centroid = pc[:,0,:]
    centroid = np.expand_dims(centroid, axis=1)
    pc = pc - centroid

    m = np.max(np.sqrt(np.sum(pc**2, axis=2)),axis = 1, keepdims=True)
    scale_inv = m #B
    pc = pc / np.expand_dims(m, axis=-1)

    for B in range(pc.shape[0]):
        x, pca_martix_inv = pca_alignment(pc[B,:,:])
        normalize_data[B, ...] = x
        martix_inv[B, ...] = pca_martix_inv

    return normalize_data, martix_inv, scale_inv

def main(args,test_data_dir):
    def log_string(str):
        logger.info(str)
        print(str)

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    '''CREATE DIR'''
    experiment_dir = Path('./log/')
    experiment_dir = experiment_dir.joinpath('path_denoise/'+ args.model)
    checkpoints_dir = experiment_dir.joinpath('checkpoints/')

    '''LOG'''
    args = parse_args()
    name_dir = test_data_dir 
    root = 'data/test_data/'

    DATA_PATH = 'data/test_data/' + name_dir + '/' 
    samples = glob(DATA_PATH+"/*.xyz")  
    samples.sort()
    #print(samples)
    batch_size = args.batch_size

    block_num = args.block_num
    path_num = args.path_num

    '''MODEL LOADING'''
    MODEL = importlib.import_module(args.model)

    denoiser = MODEL.get_model(block_num, path_num).cuda()
    analyser = MODEL.get_analyser(block_num, path_num).cuda()
 
    checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth')
    denoiser.load_state_dict(checkpoint['denoiser_model_state_dict'])
    analyser.load_state_dict(checkpoint['analyser_model_state_dict'])


    '''save dir'''
    save_dir = './data/test_data/results/' + name_dir
    if(not os.path.exists(save_dir)):
       os.makedirs(save_dir)

    for i,item in tqdm(enumerate(samples)):
        print(item)
        start_time = time.time()
        data_name = item.split('/')[-1][:-4]
        input_data = np.loadtxt(DATA_PATH + data_name + '.xyz').astype(np.float32) #

        with torch.no_grad():
            denoiser = denoiser.eval()
            analyser = analyser.eval()
            source_data = input_data[:,:3]

            path_first = None

            for iter_time in range(2):

                inputs = input_data[:,:3]
                nbrs = spatial.cKDTree(inputs)  # kd tree

                batch_num = int(inputs.shape[0] / batch_size)
                add_batch = 0 if(inputs.shape[0] % batch_size ==0)else 1

                normalize_trans_all = []
                path_all = []
                
                dist,idxs = nbrs.query(inputs,k = int(128)) #s

                input_patchs = inputs[idxs,:] # B,128,3

                normalize_input_patchs, martix_inv, scale_inv = pca_normalize(input_patchs)  # B,K,3  #B,3,3  #B
                print('time1:', time.time()-start_time)
                for i in tqdm(range(batch_num+add_batch)):
                    points = normalize_input_patchs[i*batch_size:(i+1)*batch_size,:,:] # B,128,3
                
                    points = torch.Tensor(points)
                    points = points.float().cuda()
      
                    points = points.transpose(2, 1)

                    trans_m, path_m, _ = denoiser(points,analyser,args.use_random_path)

                    trans_m = torch.cat(trans_m,axis = 0).reshape(block_num,-1,3).transpose(1,0) #B,block_num,3

                    path_m = torch.cat(path_m,axis = 0).reshape(block_num,-1).transpose(1,0)  #B,block_num

                    normalize_trans_all.append(trans_m)
                    path_all.append(path_m)

                normalize_trans_all = torch.cat(normalize_trans_all, axis = 0)  # B,6,3
                path_all = torch.cat(path_all, axis = 0)

                normalize_trans_all = normalize_trans_all.cpu().numpy().astype(np.float32).reshape(-1,block_num,3)
                path_all = path_all.cpu().numpy().astype(np.float32).reshape(-1,block_num)

                trans_all = np.matmul(martix_inv, normalize_trans_all.transpose(0,2,1)).transpose(0,2,1) # B,6,3

                trans_all = trans_all * np.expand_dims(scale_inv,axis = -1)

                outputs_all = inputs[:,:3].reshape(-1,1,3) - trans_all
            
                #scale_recover
                if(iter_time ==0):
                    path_first = path_all

                #save
                inputs_start = np.concatenate((inputs[:,:3],path_all[:,0].reshape(-1,1)),axis = -1)
                np.savetxt(save_dir+"/"+ data_name +"_input_start.xyz", inputs_start.astype(np.float32), fmt = '%.6f')

                inputs_sum = np.concatenate((inputs[:,:3],np.sum(path_all, axis = 1).reshape(-1,1)),axis = -1)
                np.savetxt(save_dir+"/"+ data_name +"_input_sum.xyz",inputs_sum.astype(np.float32),fmt = '%.6f')

                output_end = np.concatenate((outputs_all[:,-1,:],np.sum(path_first, axis = 1).reshape(-1,1)),axis = -1)
                np.savetxt(save_dir+"/"+ data_name +"_output_end.xyz",output_end.astype(np.float32),fmt = '%.6f')

                for i in range(0,block_num-1):
                    path_i = path_all[:,i+1].reshape(-1,1) 
                    output_i = outputs_all[:,i,:]
                    output_i = np.concatenate((output_i,path_i),axis = -1)
                    np.savetxt(save_dir+"/"+ data_name +"_output_" + str(i) + ".xyz",output_i.astype(np.float32),fmt = '%.6f')

                input_data = outputs_all[:,-1,:] #iter
                data_name = data_name +'_output_end'
            print('time2:', time.time()-start_time)
if __name__ == '__main__':
    args = parse_args()
    test_data_dir =  'benchmark81_20000' # test dir ['benchmark81_10000','benchmark81_20000','benchmark81_50000','kinect_fusion','kinect_v1','kinect_v2']
    main(args,test_data_dir)

