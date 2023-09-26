"""
Author: Zeyong Wei
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

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))


def parse_args():
    parser = argparse.ArgumentParser('Model')
    parser.add_argument('--model', type=str, default='model', help='model name [default: model]')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch Size during training [default: 16]')
    parser.add_argument('--epoch',  default=300, type=int, help='Epoch to run [default: 251]')
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
    parser.add_argument('--use_random_path', type=int, default=0, help='whether use random path, 0 no ,1 yes, 2 all 0, 3 all 1 ')
    parser.add_argument('--block_num', type=int, default=6, help='num of denosier block')
    parser.add_argument('--path_num', type=int, default=2, help='path num of each denosier block')

    return parser.parse_args()

def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    '''CREATE DIR'''
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    experiment_dir = Path('./log/')
    experiment_dir.mkdir(exist_ok=True)
    experiment_dir = experiment_dir.joinpath('path_denoise')
    experiment_dir.mkdir(exist_ok=True)
    if args.log_dir is None:
        experiment_dir = experiment_dir.joinpath(timestr)
    else:
        experiment_dir = experiment_dir.joinpath(args.log_dir)
    experiment_dir.mkdir(exist_ok=True)
    checkpoints_dir = experiment_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = experiment_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)

    '''LOG'''
    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, args.model))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    root = 'data/'

    TRAIN_DATASET = PatchDataset(root = root, npoints=args.npoint, split='train', normal_channel=args.normal)
    trainDataLoader = torch.utils.data.DataLoader(TRAIN_DATASET, batch_size=args.batch_size,shuffle=True, num_workers=4)
    TEST_DATASET = PatchDataset(root = root, npoints=args.npoint, split='test', normal_channel=args.normal)
    testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=args.batch_size,shuffle=False, num_workers=4)
    log_string("The number of training data is: %d" % len(TRAIN_DATASET))
    log_string("The number of test data is: %d" %  len(TEST_DATASET))

    block_num = args.block_num
    path_num = args.path_num

    '''MODEL LOADING'''
    MODEL = importlib.import_module(args.model)
    shutil.copy('models/%s.py' % args.model, str(experiment_dir))
    shutil.copy('models/pointnet_util.py', str(experiment_dir))
    shutil.copy('data_utils/DataLoader.py', str(experiment_dir))
    shutil.copy('provider.py', str(experiment_dir))
    shutil.copy('train.py', str(experiment_dir))
    shutil.copy('test.py', str(experiment_dir))

    denoiser = MODEL.get_model(block_num, path_num).cuda()
    criterion = MODEL.get_loss().cuda()

    analyser = MODEL.get_analyser(block_num, path_num).cuda()
    get_reward = MODEL.get_reward().cuda()

    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv2d') != -1:
            torch.nn.init.xavier_normal_(m.weight.data)
            torch.nn.init.constant_(m.bias.data, 0.0)
        elif classname.find('Linear') != -1:
            torch.nn.init.xavier_normal_(m.weight.data)
            torch.nn.init.constant_(m.bias.data, 0.0)

    try:
        checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth')
        denoiser.load_state_dict(checkpoint['denoiser_model_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        log_string('Use pretrain model')
    except:
        log_string('No existing model, starting training from scratch...')
        start_epoch = 0


    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            denoiser.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.decay_rate
        )
    else:
        optimizer = torch.optim.SGD(denoiser.parameters(), lr=args.learning_rate, momentum=0.9)

    if args.optimizer2 == 'Adam':
        optimizer2 = torch.optim.Adam(
            analyser.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.decay_rate
        )
    else:
        optimizer2 = torch.optim.SGD(analyser.parameters(), lr=args.learning_rate, momentum=0.9)


    LEARNING_RATE_CLIP = 1e-6

    best_loss_denoise = 999999
    global_epoch = 0

    for epoch in range(start_epoch,args.epoch):
        log_string('Epoch %d (%d/%s):' % (global_epoch, epoch, args.epoch))
        '''Adjust learning rate and BN momentum'''
        lr = max(args.learning_rate * (args.lr_decay ** (epoch // args.step_size)), LEARNING_RATE_CLIP)
        log_string('Learning rate:%f' % lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        if(args.use_random_path == 0):
            for param_group in optimizer2.param_groups:
                param_group['lr'] = lr

        num_batches = len(trainDataLoader)

        '''learning one epoch'''
        loss_sum = 0
        reward_sum = 0

        for i, data in tqdm(enumerate(trainDataLoader), total=len(trainDataLoader), smoothing=0.9):
            inputs, target, label = data
            cur_batch_size, NUM_POINT, _ = inputs.size() #B,N,3
            
            points = inputs.data.numpy()
            target = target.data.numpy()
            label = label.data.numpy()

            points = provider.random_point_dropout(points)

            source = points[:,0,:]   #B,3

            points = torch.Tensor(points)
            source = torch.Tensor(source)
            target = torch.Tensor(target)
            label = torch.Tensor(label)

            points, source, target, label = points.float().cuda(), source.float().cuda(), target.float().cuda(), label.float().cuda()


            points = points.transpose(2, 1)

            if(args.use_random_path == 1):
                optimizer.zero_grad()
                denoiser = denoiser.train()
                analyser = analyser.eval()

                trans_m, path_m, path_maxprob_m = denoiser(points,analyser,args.use_random_path)
          
                trans = trans_m[-1].reshape(cur_batch_size, 3)
                points_denoise = source - trans

                loss = criterion(source, target, trans_m, None, None, 1)

                loss.backward()
                optimizer.step()
                loss_sum += loss

            elif(args.use_random_path == 0):
                
                optimizer.zero_grad()
                denoiser = denoiser.train() 

                optimizer2.zero_grad()
                analyser = analyser.train()

                trans_m, path_m, path_maxprob_m = denoiser(points,analyser,args.use_random_path)
            
                trans = trans_m[-1].reshape(cur_batch_size, 3)
                points_denoise = source - trans

                loss = criterion(source, target, trans_m, path_m, path_maxprob_m, 0)
                reward = get_reward(source, target, label, trans_m, path_m, path_maxprob_m)

                loss.backward(retain_graph=True)
                loss_sum += loss

                reward.backward()
                reward_sum += reward

                optimizer.step() 
                optimizer2.step()
                
        if(args.use_random_path == 1):
            log_string('Training mean loss: %f' % (loss_sum / num_batches))
        elif(args.use_random_path == 0):
            log_string('Training mean loss: %f, Training mean reward: %f' % (loss_sum / num_batches,reward_sum / num_batches))

        with torch.no_grad():
            test_metrics = {}
            cur_mean_loss = []
            cur_mean_reward = []

            for batch_id, data in tqdm(enumerate(testDataLoader), total=len(testDataLoader), smoothing=0.9):
                inputs, target, label = data
                cur_batch_size, NUM_POINT, _ = inputs.size() #B,N,3

                points = inputs.data.numpy()
                target = target.data.numpy()
                label = label.data.numpy()

                source = points[:,0,:]   #B,3

                points = torch.Tensor(points)
                source = torch.Tensor(source)
                target = torch.Tensor(target)
                label = torch.Tensor(label)

                points, source, target, label = points.float().cuda(), source.float().cuda(), target.float().cuda(), label.float().cuda()
      
                points = points.transpose(2, 1)

                denoiser = denoiser.eval()

                if(args.use_random_path == 0):
                    analyser = analyser.eval()

                trans_m, path_m, path_maxprob_m = denoiser(points,analyser,args.use_random_path)
            
                trans = trans_m[-1].reshape(cur_batch_size,3)
                points_denoise = source - trans

                if(args.use_random_path == 1):
                    loss = criterion(source, target, trans_m, None, None, 1)
                    cur_mean_loss.append(loss.item())
                elif(args.use_random_path == 0):
                    loss = criterion(source, target, trans_m, path_m, path_maxprob_m, 0)
                    reward = get_reward(source, target, label, trans_m, path_m, path_maxprob_m)

                    cur_mean_loss.append(loss.item())
                    
                    cur_mean_reward.append(reward.item())

        if(args.use_random_path == 1):           
            test_metrics['loss_denoise'] = np.mean(cur_mean_loss)
            log_string('Epoch %d test loss_denoise: %f' % (epoch, test_metrics['loss_denoise']))
        elif(args.use_random_path == 0):
            test_metrics['loss_denoise'] = np.mean(cur_mean_loss)
            test_metrics['reward_denoise'] = np.mean(cur_mean_reward) 
            log_string('Epoch %d test loss_denoise: %f, test reward_denoise: %f' % (epoch, test_metrics['loss_denoise'],test_metrics['reward_denoise']))

        if (epoch%10 == 0):
            logger.info('Save model...')
            savepath = str(checkpoints_dir) + '/model_'+ str(epoch) +'.pth'
            log_string('Saving at %s'% savepath)
            state = {
                'epoch': epoch,
                'denoiser_model_state_dict': denoiser.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'analyser_model_state_dict': analyser.state_dict(),
                'optimizer2_state_dict': optimizer2.state_dict(),       
            }
            torch.save(state, savepath)
            log_string('Saving model....')

        if test_metrics['loss_denoise'] < best_loss_denoise:
            best_loss_denoise = test_metrics['loss_denoise']
            if (True):
                logger.info('Save model...')
                savepath = str(checkpoints_dir) + '/model_'+ str(epoch) +'.pth'
                savepath2 = str(checkpoints_dir) + '/best_model' +'.pth'
                log_string('Saving at %s'% savepath)
                state = {
                    'epoch': epoch,
                    'denoiser_model_state_dict': denoiser.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'analyser_model_state_dict': analyser.state_dict(),
                    'optimizer2_state_dict': optimizer2.state_dict(), 
                }
                if(epoch > 1):
                    torch.save(state, savepath)
                torch.save(state, savepath2)
                log_string('Saving model....')
        if test_metrics['loss_denoise'] < best_loss_denoise:
            best_loss_denoise = test_metrics['loss_denoise']
        if(args.use_random_path == 1): 
            log_string('Best loss_denoise is: %.6f'%(best_loss_denoise))
        elif(args.use_random_path == 0): 
            log_string('Best loss_denoise is: %.6f, Best reward_denoise is: %.6f'%(best_loss_denoise, test_metrics['reward_denoise']))
        global_epoch+=1

if __name__ == '__main__':
    args = parse_args()
    main(args)

