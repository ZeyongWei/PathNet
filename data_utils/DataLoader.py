# *_*coding:utf-8 *_*
import os
import warnings
import numpy as np
import h5py
from torch.utils.data import Dataset
from data_utils.Pointfilter_Utils import pca_alignment

warnings.filterwarnings('ignore')

def pca_normalize(pc,pc2):
    normalize_data = np.zeros(pc.shape, dtype=np.float32)
    normalize_data2 = np.zeros(pc2.shape, dtype=np.float32)

    centroid = pc[:,0,:]
    centroid = np.expand_dims(centroid, axis=1)
    pc = pc - centroid
    pc2 = pc2 - centroid

    m = np.max(np.sqrt(np.sum(pc**2, axis=2)),axis = 1, keepdims=True)
    pc = pc / np.expand_dims(m, axis=-1)
    pc2 = pc2 / np.expand_dims(m, axis=-1)

    for B in range(pc.shape[0]):
        x, pca_martix_inv = pca_alignment(pc[B,:,:])
        x2 = np.array(np.linalg.inv(pca_martix_inv) * np.matrix(pc2[B,:,:].T)).T
        normalize_data[B, ...] = x
        normalize_data2[B, ...] = x2

    return normalize_data, normalize_data2

class PatchDataset(Dataset):
    def __init__(self,root = './data/', npoints=128, split='train', class_choice=None, normal_channel=False):
        self.npoints = npoints
        self.root = root

        self.catfile = os.path.join(self.root, 'train _data.hdf5')

        f = h5py.File(self.catfile,'r')

        self.inputs = f["inputs"][:]  #B, N ,3
        self.target = f["target"][:]
        self.label = f["label"][:]

        self.inputs,self.target= pca_normalize(self.inputs,self.target)
 
        self.label = self.label //2.3

        idx = np.arange(0,self.inputs.shape[0])
        np.random.seed(1111)
        np.random.shuffle(idx)
        self.inputs = self.inputs[idx][:,:,:3]
        self.target = self.target[idx][:,:,:3]
        self.label = self.label[idx][:]

        sample_size = int(self.inputs.shape[0] * 0.8)
        if(split == 'train'):
            self.inputs = self.inputs[:sample_size]
            self.target = self.target[:sample_size]
            self.label = self.label[:sample_size]
        elif(split == 'test'):
            self.inputs = self.inputs[sample_size:]
            self.target = self.target[sample_size:]
            self.label = self.label[:sample_size]

        print('The size of %s inputs is %d'%(split,self.inputs.shape[0]))


        self.seg_classes = {'circle': [0,1]}

        self.cache = {}  
        self.cache_size = 1000


    def __getitem__(self, index):
        if index in self.cache:
            inputs, target, label = self.cache[index]
        else:
            inputs = self.inputs[index].astype(np.float32)  #N,3
            target = self.target[index].astype(np.float32)
            label = self.label[index].astype(np.float32)

            if len(self.cache) < self.cache_size:
                self.cache[index] = (inputs, target, label)

        return inputs, target, label

    def __len__(self):
        return self.inputs.shape[0]



