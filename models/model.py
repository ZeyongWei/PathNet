import torch.nn as nn
import torch.utils.data
import torch
import torch.nn.functional as F
import numpy as np
from einops import rearrange, repeat

class Analyser_Block(nn.Module):
    def __init__(self, path_num = 3):
        super(Analyser_Block, self).__init__()
        self.path_num = path_num

        #encoder
        self.mlp_convs1 = nn.Conv1d(512,512,1)
        self.mlp_convs2 = nn.Conv1d(512,512,1)

        self.mlp_bns1 = nn.BatchNorm1d(512)
        self.mlp_bns2 = nn.BatchNorm1d(512)
        
        # decoder
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, self.path_num)

        self.bn1 = nn.BatchNorm1d(256)
        self.bn2 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.4)
        self.drop2 = nn.Dropout(0.4)

    def forward(self, f1):
        #encoder
        f1 = F.relu(self.mlp_bns1(self.mlp_convs1(f1)))
        f1 = F.relu(self.mlp_bns2(self.mlp_convs2(f1)))

        #decoder
        f1_max = torch.max(f1, axis = 2)[0]

        path_f1 = self.drop1(F.relu(self.bn1(self.fc1(f1_max))))
        path_f1 = self.drop2(F.relu(self.bn2(self.fc2(path_f1))))
        path_f1 = self.fc3(path_f1)  #B,3
        path_f1 = F.softmax(path_f1,dim=1)

        return path_f1

class get_analyser(nn.Module):
    def __init__(self, block_num = 3, path_num = 3):
        super(get_analyser, self).__init__()
        self.block_num = block_num
        self.path_num = path_num
        #Analyser_Block
        self.analyser_blocks = nn.ModuleList()
        for i in range(self.block_num):
            self.analyser_blocks.append(Analyser_Block(self.path_num))

    def forward(self, f1, ab_i = 0):

        path_f1 = self.analyser_blocks[ab_i](f1)

        return path_f1

class Path_Block(nn.Module):
    def __init__(self, path_num = 3):
        super(Path_Block, self).__init__()
        self.path_num = path_num

        #path-head
        self.mlp_convs_ph1 = nn.Conv1d(512,512,1)
        self.mlp_convs_ph2 = nn.Conv1d(512,512,1)

        self.mlp_bns_ph1 = nn.BatchNorm1d(512)
        self.mlp_bns_ph2 = nn.BatchNorm1d(512)

        #path0
        #pass

        #path1
        self.mlp_convs_p11 = nn.Conv1d(512,256,1)
        self.mlp_convs_p12 = nn.Conv1d(512,512,1)

        self.mlp_bns_p11 = nn.BatchNorm1d(256)
        self.mlp_bns_p12 = nn.BatchNorm1d(512)

        #path2
        self.mlp_convs_p21 = nn.Conv1d(512,256,1)
        self.mlp_convs_p22 = nn.Conv1d(512,512,1)

        self.mlp_bns_p21 = nn.BatchNorm1d(256)
        self.mlp_bns_p22 = nn.BatchNorm1d(512)

        self.mlp_convs_p23 = nn.Conv1d(512,256,1)
        self.mlp_convs_p24 = nn.Conv1d(512,512,1)

        self.mlp_bns_p23 = nn.BatchNorm1d(256)
        self.mlp_bns_p24 = nn.BatchNorm1d(512)

        #decoder
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 3)

        self.bn1 = nn.BatchNorm1d(256)
        self.bn2 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.4)
        self.drop2 = nn.Dropout(0.4)

    def forward(self, f1, path_f1):
        #path_f1.shape #B

        #path_head
        f1_temp = F.relu(self.mlp_bns_ph1(self.mlp_convs_ph1(f1)))  #B, d, N
        f1 = f1 + F.relu(self.mlp_bns_ph2(self.mlp_convs_ph2(f1_temp)))  #B, d, N

        #path_chose

        if(self.path_num == 2):

            f1_out = f1
            idx = torch.where(path_f1==1)
            if(idx[0].shape[0]>0):
                f1_out[idx] = self.path1(f1_out[idx])
        
        #decoder
        t1 = self.path_denoise(f1_out)

        return f1_out, t1

    def path0(self,f1):

        #pass
        return f1

    def path1(self,f1):

        f1_temp = F.relu(self.mlp_bns_p11(self.mlp_convs_p11(f1)))
        f1_temp_max = torch.max(f1_temp, axis = 2)[0]
        f1_temp_max = f1_temp_max.unsqueeze(-1).repeat(1,1,f1_temp.shape[-1])
        f1_cat = torch.cat((f1_temp,f1_temp_max),1)
        f1 = f1 + F.relu(self.mlp_bns_p12(self.mlp_convs_p12(f1_cat)))

        return f1

    def path2(self,f1):
        f1_temp = F.relu(self.mlp_bns_p21(self.mlp_convs_p21(f1)))
        f1_temp_max = torch.max(f1_temp, axis = 2)[0]
        f1_temp_max = f1_temp_max.unsqueeze(-1).repeat(1,1,f1_temp.shape[-1])
        f1_temp = torch.cat((f1_temp,f1_temp_max),1)
        f2 = f1 + F.relu(self.mlp_bns_p22(self.mlp_convs_p22(f1_temp)))

        f1_temp = F.relu(self.mlp_bns_p23(self.mlp_convs_p23(f2)))
        f1_temp_max = torch.max(f1_temp, axis = 2)[0]
        f1_temp_max = f1_temp_max.unsqueeze(-1).repeat(1,1,f1_temp.shape[-1])
        f1_temp = torch.cat((f1_temp,f1_temp_max),1)
        f1 = f1 + F.relu(self.mlp_bns_p24(self.mlp_convs_p24(f1_temp)))

        return f1

    def path_denoise(self, f1):

        f1_max = torch.max(f1, axis = 2)[0]

        t1 = self.drop1(F.relu(self.bn1(self.fc1(f1_max))))
        t1 = self.drop2(F.relu(self.bn2(self.fc2(t1))))
        t1 = self.fc3(t1) #B,3

        return t1

class get_model(nn.Module):
    def __init__(self, block_num = 3, path_num = 3):
        super(get_model, self).__init__()
        channel = 3
        self.block_num = block_num
        self.path_num = path_num

        #encoders
        self.mlp = np.array([64,128,256,512])

        self.mlp_convs = nn.ModuleList()

        self.mlp_bns = nn.ModuleList()

        last_channel = channel
        for i in range(self.mlp.shape[0]):
            self.mlp_convs.append(nn.Conv1d(last_channel, self.mlp[i], 1))
            self.mlp_bns.append(nn.BatchNorm1d(self.mlp[i]))

            last_channel = self.mlp[i]

        #path_blocks
        self.pbs = nn.ModuleList()
        for i in range(self.block_num):
            self.pbs.append(Path_Block(self.path_num))


    def forward(self, x, analyser, use_random_path = 0):
        B, _, N = x.shape

        #encoder
        x = x - x[:,:,0:1]  #B,3,N

        f0 = x
        for i in range(self.mlp.shape[0]):
            f0 = F.relu(self.mlp_bns[i](self.mlp_convs[i](f0)))

        feature_m = []
        path_maxprob_m = []
        path_m = []
        trans_m = []

        f1 = f0
        for pb_i in range(self.block_num):
            #path_analyser
            if(use_random_path == 1):
                path_prob_f1 = torch.rand(B,self.path_num)
                path_prob_f1 = F.softmax(path_prob_f1,-1)

            elif(pb_i < self.block_num and use_random_path == 0): 
                path_prob_f1 = analyser(f1, pb_i)

            else:
                print("error")
                return 0,0,0

            path_maxprob_f1, path_f1 = torch.max(path_prob_f1,axis = -1)#B
 
            #path_block
            f1, t1 = self.pbs[pb_i](f1,path_f1)

            #recode
            feature_m.append(f1)
            path_maxprob_m.append(path_maxprob_f1)
            path_m.append(path_f1)
            trans_m.append(t1) 

        return trans_m, path_m, path_maxprob_m

class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()
        
    def forward(self, source, target, trans_m, path_m, path_maxprob_m, all_stage = 1):
        loss_m = []
        if(all_stage == 1):
            for i in range(len(trans_m)):
                trans_i = trans_m[i]
                points_denoise_i = source - trans_i
                loss_i = self.catculate_loss(points_denoise_i,target)
                loss_m.append(loss_i)
        elif(all_stage == 0):
            trans_end = trans_m[-1]
            points_denoise_end = source - trans_end
            loss_end = self.catculate_loss(points_denoise_end,target)
            loss_m.append(loss_end)

        loss = loss_m[-1]
        for i in range(len(loss_m)-1):
            loss = loss + 0.1*loss_m[i]

        return loss

    def catculate_loss(self,p1,p2): #B,3 #B,N,3
        dist = torch.sum((p1.unsqueeze(1) - p2)**2,axis = -1) #B,N

        dist_min = torch.min(dist,axis = -1)[0]  #B
        dist_max = torch.max(dist,axis = -1)[0]  #B

        loss_1 = torch.mean(dist_min)
        loss_2 = torch.mean(dist_max)

        loss = 0.99 * loss_1 + 0.01* loss_2

        return loss


class get_reward(nn.Module):
    def __init__(self):
        super(get_reward, self).__init__()
        self.reward_add = 0
        self.itt = 0
        
    def forward(self, source, target, label, trans_m, path_m, path_maxprob_m):

        trans_end = trans_m[-1]
        points_denoise = source - trans_end    

        loss_start = self.catculate_loss2(source,target) # B
        loss_end = self.catculate_loss2(points_denoise,target) # B

        loss_dt = loss_end - loss_start  # B

        p = torch.Tensor([0.002]).cuda()  #  penalty p 0.0002

        L0 = torch.Tensor([0.4]).cuda()   # threshold max(L)  0.4
        d = loss_end / L0   #

        d = torch.where(d>1.0,torch.ones_like(d),d)
        l = torch.exp((-1.0)*(label))
        lammda = 0.05

        path_end = path_m[-1].cuda()
        #reward_end = (-1) * p * path_end + (-1) * d * loss_dt  #B noise awareness
        reward_end = (-1) * p * path_end + (-1) * (d + lammda * l)* loss_dt  #B geometric awareness

        rewards = []
        for i in range(len(path_m)-1):
            path_i = path_m[i].cuda() #B
            reward = (-1) * p * path_i   #B
            rewards.append(reward.unsqueeze(-1))
        rewards.append(reward_end.unsqueeze(-1))
        rewards = torch.cat(rewards,axis = -1)
 
        loss = self.catculate_rewards_loss(rewards, path_maxprob_m)

        loss = torch.mean(loss)
        
        return loss

    def catculate_loss2(self,p1,p2): #B,3 #B,N,3
        dist = torch.sum((p1.unsqueeze(1) - p2)**2,axis = -1) #B,N

        dist_min = torch.min(dist,axis = -1)[0]  #B
        dist_max = torch.max(dist,axis = -1)[0]  #B

        loss_1 = dist_min
        loss_2 = dist_max

        loss = 0.99 * loss_1 + 0.01* loss_2

        return loss

    def catculate_rewards_loss(self,rewards, path_maxprob_m):
        R = torch.zeros(rewards.shape[0]).cuda()
        loss = 0
        gamma = 0.99 ###########
        for i in reversed(range(rewards.shape[1])): 
            R = gamma * R + torch.log(path_maxprob_m[i]) * (rewards[:,i] + 0.02) 
            loss = loss - R

            #R = gamma * R + rewards[:,i]
            #loss = loss - torch.log(path_maxprob_m[i])*(R + 0.02)
        loss = loss / rewards.shape[1]

        return loss


