'''
Seokju Lee, 2018.04.11
PyTorch implementation of "Co-domain Embedding using Deep Quadruplet Networks for Unseen Traffic Sign Recognition (AAAI-18)"
Base codes from "https://github.com/SeokjuLee/SiameseNet-TS"

v1: QuadNetSingle (IdsiaNet, single tower)
v2: QuadNet (IdsiaNet, two tower)

python main.py -a QuadNetSingle
python main.py -a QuadNetSingle --evaluate --pretrained /media/rcv/SSD1/git/QuadNet/gtsrb_data/Wed-Apr-11-17:12/100epochs,b50,lr0.0005/model_best.pth.tar

python main.py
python main.py --evaluate --pretrained /media/rcv/SSD1/git/QuadNet/gtsrb_data/Wed-Apr-11-19:49/100epochs,b50,lr0.0005/model_best.pth.tar

python main.py --dataset tt100k_data


ps aux | grep python
'''

import argparse
import os
import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader,Dataset
import matplotlib.pyplot as plt
import torchvision.utils
import numpy as np
import random
from PIL import Image
import torch
from torch.autograd import Variable
import PIL.ImageOps    
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from model import QuadNet
from model import QuadNetSingle
from loss import ContrastiveLoss
from loss import HingeMLoss
import data_transform
import datetime
import datasets
import csv
import time
import shutil
import progressbar

import pdb



# random.seed(1)
dataset_names = sorted(name for name in datasets.__all__)

parser = argparse.ArgumentParser(description='PyTorch SiameseNet Training on several datasets')
parser.add_argument('--lr', '--learning-rate', default=0.0005, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                    help='number of data loading workers (default: 0)')
parser.add_argument('-b', '--batch-size', default=50, type=int,
                    metavar='N', help='mini-batch size (default: 8)')
parser.add_argument('-e', '--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run (default: 300')
parser.add_argument('--dataset', metavar='DATASET', default='gtsrb_data',
                    choices=dataset_names,
                    help='dataset type : ' +
                        ' | '.join(dataset_names) +
                        ' (default: gtsrb_data)')
parser.add_argument('--pretrained', dest='pretrained', default = None,
                    help='path to pre-trained model')
parser.add_argument('--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--log-summary', default = 'progress_log_summary.csv',
                    help='csv where to save per-epoch train and test stats')
parser.add_argument('--log-full', default = 'progress_log_full.csv',
                    help='csv where to save per-gradient descent train stats')
parser.add_argument('--data-parallel', default=None,
                    help='Use nn.DataParallel() model')
parser.add_argument('--print-freq', '-p', default=100, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--arch','-a', default='QuadNet',
                    help='select architecture')


args = parser.parse_args()

if args.dataset == 'gtsrb_data':
    ### GTSRB
    class Config():
        base_path   = "/media/rcv/SSD1/Logo_oneshot/GTSRB"
        tr_im_path  = "/media/rcv/SSD1/Logo_oneshot/GTSRB/Experiment02-22-43/train_impaths.txt"
        tr_gt_path  = "/media/rcv/SSD1/Logo_oneshot/GTSRB/Experiment02-22-43/train_imclasses.txt"
        te_im_path  = "/media/rcv/SSD1/Logo_oneshot/GTSRB/Experiment02-22-43/test_impaths.txt"
        te_gt_path  = "/media/rcv/SSD1/Logo_oneshot/GTSRB/Experiment02-22-43/test_imclasses.txt"
        tr_tmp_path = "/media/rcv/SSD1/Logo_oneshot/GTSRB/GTSRB_template_ordered"
        te_tmp_path = "/media/rcv/SSD1/Logo_oneshot/GTSRB/GTSRB_template_ordered"

elif args.dataset == 'tt100k_data':
    ### TT100K
    class Config():
        base_path   = "/media/rcv/SSD1/Logo_oneshot/TT100K"
        tr_im_path  = "/media/rcv/SSD1/Logo_oneshot/TT100K/exp02_exist_classes_only/train_impaths.txt"
        tr_gt_path  = "/media/rcv/SSD1/Logo_oneshot/TT100K/exp02_exist_classes_only/train_imclasses.txt"
        te_im_path  = "/media/rcv/SSD1/Logo_oneshot/TT100K/exp02_exist_classes_only/val_impaths.txt"
        te_gt_path  = "/media/rcv/SSD1/Logo_oneshot/TT100K/exp02_exist_classes_only/val_imclasses.txt"
        tr_tmp_path = "/media/rcv/SSD1/Logo_oneshot/TT100K/TT100K_template_ordered"
        te_tmp_path = "/media/rcv/SSD1/Logo_oneshot/TT100K/TT100K_template_ordered"


BEST_TEST_LOSS = -1


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, os.path.join(save_path,filename))
    if is_best:
        shutil.copyfile(os.path.join(save_path,filename), os.path.join(save_path,'model_best.pth.tar'))


def imshow(img,text=None,should_save=False):
    npimg = img.numpy()
    plt.axis("off")
    if text:
        plt.text(60, 8, text, style='italic',fontweight='bold',
            bbox={'facecolor':'white', 'alpha':0.8, 'pad':10})
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.ion()
    plt.show()    

def show_plot(iteration,loss):
    plt.plot(iteration,loss)
    plt.show()



def main():    
    global args, BEST_TEST_LOSS, save_path
    
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])    
    input_transform = transforms.Compose([
                data_transform.PILScale((48,48)),
                transforms.ToTensor(),
                # normalize
        ])

    print("=> fetching image/label pairs in '{}'".format(args.dataset))
    train_set, test_set = datasets.__dict__[args.dataset](
        Config.base_path,
        Config.tr_im_path, 
        Config.tr_gt_path, 
        Config.te_im_path, 
        Config.te_gt_path, 
        Config.tr_tmp_path,
        Config.te_tmp_path,
        transform=input_transform,
        split=100,
        should_invert=False,
    )

    print('{} samples found, {} train samples and {} test samples '.format(len(test_set)+len(train_set),
                                                                           len(train_set),
                                                                           len(test_set)))


    ### Data visualization #####################
    # vis_dataloader = DataLoader(train_set,
    #                         shuffle=True,
    #                         num_workers=args.workers,
    #                         batch_size=args.batch_size)
    # dataiter = iter(vis_dataloader)
    # example_batch = next(dataiter)
    # concatenated = torch.cat((example_batch[0],example_batch[2]),0)
    # imshow(torchvision.utils.make_grid(concatenated))
    # print(example_batch[4].numpy())
    # pdb.set_trace()
    #############################################


    train_loader = DataLoader(train_set,
                        shuffle=True,
                        num_workers=args.workers,
                        batch_size=args.batch_size)
    test_loader = DataLoader(test_set, 
                        shuffle=False, 
                        num_workers=args.workers, 
                        batch_size=args.batch_size)

    if args.architecture == 'QuadNet':
        net = QuadNet().cuda()
    if args.architecture == 'QuadNetSingle':
        net = QuadNetSingle().cuda()

    if args.pretrained:
        print("=> Use pre-trained model")
        weights = torch.load(args.pretrained)
        net.load_state_dict(weights['state_dict'])
    else:
        print("=> Randomly initialize model")
        net.init_weights()

    
    criterion = HingeMLoss()
    optimizer = optim.Adam( net.parameters(), lr=args.lr )

    if args.data_parallel:
        net = torch.nn.DataParallel(net).cuda()


    if args.evaluate:
        eval(args.dataset, test_set, net, input_transform)


    ### Visualize testset with pretrained model #####################
    # while True:
    #     test_dataloader = DataLoader(test_set, shuffle=True, num_workers=args.workers, batch_size=1)
    #     dataiter = iter(test_dataloader)

    #     for i in range(1):
    #         ra, rb, ta, tb, _, _ = next(dataiter)
    #         concatenated = torch.cat((ra,rb),0)
    #         ra = Variable(ra).cuda()
    #         rb = Variable(rb).cuda()
    #         ta = Variable(ta).cuda()
    #         tb = Variable(tb).cuda()

    #         RA,RB,TA,TB = net(ra,rb,ta,tb)
    #         euclidean_distance = F.pairwise_distance(RA, RB)
    #         imshow(torchvision.utils.make_grid(concatenated),'Dissimilarity: {:.2f}'.format(euclidean_distance.cpu().data.numpy()[0][0]))
    #     pdb.set_trace()
    ##################################################################


    save_path = '{}epochs,b{},lr{}'.format(
        args.epochs,
        args.batch_size,
        args.lr)
    timestamp = datetime.datetime.now().strftime("%a-%b-%d-%H:%M")
    save_path = os.path.join(timestamp,save_path)
    save_path = os.path.join(args.dataset,save_path)
    print('=> will save everything to {}'.format(save_path))
    if not os.path.exists(save_path):
            os.makedirs(save_path)


    with open(os.path.join(save_path,args.log_summary), 'w') as csvfile:    # save every epoch
        writer = csv.writer(csvfile, delimiter='\t')
        writer.writerow(['tr_loss','te_loss'])
    with open(os.path.join(save_path,args.log_full), 'w') as csvfile:       # save every iter
        writer = csv.writer(csvfile, delimiter='\t')
        writer.writerow(['tr_loss_iter'])


    for epoch in range(0, args.epochs):
        adjust_learning_rate(optimizer, epoch)

        train_loss = train(train_loader, net, criterion=criterion, optimizer=optimizer, epoch=epoch)
        test_loss = test(test_loader, net, criterion=criterion, epoch=epoch)

        if BEST_TEST_LOSS < 0:
            BEST_TEST_LOSS = test_loss
        is_best = test_loss < BEST_TEST_LOSS
        BEST_TEST_LOSS = min(test_loss, BEST_TEST_LOSS)



        ### Save checkpoints
        if args.data_parallel:
            save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': net.module.state_dict(),  # args.data_parallel = True
                    'BEST_TEST_LOSS': BEST_TEST_LOSS,
                    }, is_best
                )
        else:
            save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': net.state_dict(),         # args.data_parallel = False
                    'BEST_TEST_LOSS': BEST_TEST_LOSS,
                    }, is_best
                )
        if (epoch+1)%10 == 0:
            ckptname = 'ckpt_e%04d.pth.tar' %(epoch+1)
            if args.data_parallel:
                torch.save({
                    'epoch': epoch + 1,
                    'state_dict': net.module.state_dict(),  # args.data_parallel = True
                    'BEST_TEST_LOSS': BEST_TEST_LOSS,
                }, os.path.join(save_path,ckptname))
            else:
                torch.save({
                    'epoch': epoch + 1,
                    'state_dict': net.state_dict(),         # args.data_parallel = False
                    'BEST_TEST_LOSS': BEST_TEST_LOSS,
                }, os.path.join(save_path,ckptname))

        ### Save epoch logs
        with open(os.path.join(save_path,args.log_summary), 'a') as csvfile:
            writer = csv.writer(csvfile, delimiter='\t')
            writer.writerow([train_loss, test_loss])



def train(train_loader, net, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    net.train()
    end = time.time()

    for i, data in enumerate(train_loader, 0):
        # if i>10: break;
        ra, rb, ta, tb, la, lb = data
        ra, rb, ta, tb = Variable(ra).cuda(), Variable(rb).cuda(), Variable(ta).cuda(), Variable(tb).cuda()
        la, lb = Variable(la).cuda(), Variable(lb).cuda()
        data_time.update(time.time() - end)

        # Assign labels: push (different class) and pull (same class)
        push = Variable(torch.FloatTensor([1])).cuda().resize(1,1).expand(ra.size(0),1)
        pull = Variable(torch.FloatTensor([0])).cuda().resize(1,1).expand(ra.size(0),1)

        RA, RB, TA, TB = net(ra,rb,ta,tb)
        # pdb.set_trace()
        loss_TATB = criterion(TA,TB,push)
        loss_TARA = criterion(TA,RA,pull)
        loss_TBRB = criterion(TB,RB,pull)
        loss_TARB = criterion(TA,RB,push)
        loss_TBRA = criterion(TB,RA,push)
        loss = loss_TATB + loss_TARA + loss_TBRB + loss_TARB + loss_TBRA
        losses.update(loss.data[0], ra.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # if epoch > 2 and i % args.print_freq and loss.data[0] > 3 == 0:
        #     ED = F.pairwise_distance(output1, output2)
        #     imgVisCat = torch.cat((img0.data.cpu(), img1.data.cpu()),0)
        #     imshow(torchvision.utils.make_grid(imgVisCat))
        #     pdb.set_trace()

        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('[{0}|{1}/{2}] '
                  'loss:{loss.val:.3f}({loss.avg:.3f})  '
                  'Batch time: {batch_time.val:.3f}({batch_time.avg:.3f})  '
                  'Data time: {data_time.val:.3f}({data_time.avg:.3f})'.format(
                   epoch, i, len(train_loader), 
                   loss=losses,
                   batch_time=batch_time, data_time=data_time))
        with open(os.path.join(save_path,args.log_full), 'a') as csvfile:
            writer = csv.writer(csvfile, delimiter='\t')
            writer.writerow([loss.data[0]])

    return losses.avg


def test(test_loader, net, criterion, epoch):
    losses = AverageMeter()

    net.eval()

    for i, data in enumerate(test_loader, 0):
        # if i>10: break;
        ra, rb, ta, tb, la, lb = data
        ra, rb, ta, tb = Variable(ra).cuda(), Variable(rb).cuda(), Variable(ta).cuda(), Variable(tb).cuda()
        la, lb = Variable(la).cuda(), Variable(lb).cuda()

        # Assign labels: push (different class) and pull (same class)
        push = Variable(torch.FloatTensor([1])).cuda().resize(1,1).expand(ra.size(0),1)
        pull = Variable(torch.FloatTensor([0])).cuda().resize(1,1).expand(ra.size(0),1)

        RA, RB, TA, TB = net(ra,rb,ta,tb)
        # pdb.set_trace()
        loss_TATB = criterion(TA,TB,push)
        loss_TARA = criterion(TA,RA,pull)
        loss_TBRB = criterion(TB,RB,pull)
        loss_TARB = criterion(TA,RB,push)
        loss_TBRA = criterion(TB,RA,push)
        loss = loss_TATB + loss_TARA + loss_TBRB + loss_TARB + loss_TBRA
        losses.update(loss.data[0], ra.size(0))


        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]  '
                  'loss: {loss.val:.3f}({loss.avg:.3f})'.format(
                   i, len(test_loader),
                   loss=losses))

    print(' * loss: {loss.avg:.3f}\t'.format(loss=losses))

    return losses.avg




def eval(db, test_set, net, input_transform):
    net.eval()

    test_loader = DataLoader(test_set, 
                        shuffle=False, 
                        num_workers=args.workers, 
                        batch_size=400)

    tmp_list = sorted( os.listdir(Config.te_tmp_path) )
    tmp = []
    for i in range(len(tmp_list)):
        img = Image.open( os.path.join(Config.te_tmp_path, tmp_list[i]) )
        img = Variable(input_transform(img)).cuda()
        tmp.append(img)

    result_table = []
    bar = progressbar.ProgressBar(max_value=len(test_loader))
    for i, data in enumerate(test_loader, 0):
        bar.update(i)
        ra, _, _, _, la, _ = data
        ra, la = Variable(ra).cuda(), Variable(la).cuda()
        ED = np.zeros( (ra.size(0), len(tmp_list)) )

        for tt in range(len(tmp_list)):
            ta = tmp[tt].unsqueeze(0).repeat(ra.size(0),1,1,1)
            RA, _, TA, _ = net(ra, ra, ta, ta)
            for bs in range(ra.size(0)):
                ED[bs][tt] = F.pairwise_distance(RA[bs].unsqueeze(0), TA[bs].unsqueeze(0)).cpu().data.numpy()[0][0]

        for bs in range(ra.size(0)):
            if db == 'gtsrb_data':
                result_table.append([int(la.cpu().data.numpy()[bs][0]), ED[bs].argmin()])
            elif db == 'tt100k_data':
                result_table.append([int(la.cpu().data.numpy()[bs][0]), ED[bs].argmin()+1])

    results = np.array(result_table)


    if db == 'gtsrb_data':
        seenList = [1,2,3,4,5,7,8,9,10,11,12,13,14,15,17,18,25,26,31,33,35,38]
        unseenList = [0,6,16,19,20,21,22,23,24,27,28,29,30,32,34,36,37,39,40,41,42]
    elif db == 'tt100k_data':
        seenList = range(1, 25)
        unseenList = range(25, 35)

    seenScore = np.zeros(len(seenList))
    unseenScore = np.zeros(len(unseenList))

    for i in range(len(seenList)):
        idx = np.where(results[:,0]==seenList[i])[0]
        num_wrong = np.count_nonzero(results[idx,0] - results[idx,1])
        seenScore[i] = float((len(idx) - num_wrong)) / float(len(idx))

    for i in range(len(unseenList)):
        idx = np.where(results[:,0]==unseenList[i])[0]
        num_wrong = np.count_nonzero(results[idx,0] - results[idx,1])
        unseenScore[i] = float((len(idx) - num_wrong)) / float(len(idx))

    print('seen:', seenScore.mean(), 'unseen:', unseenScore.mean())
    pdb.set_trace()

    


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count



def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 2 periodically"""
    if (epoch+1) % 10 == 0:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr']/2



if __name__ == '__main__':
    main()