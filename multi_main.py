# -*- coding: utf-8 -*-
import os
from sys import argv

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import PIL

import torch
import torch.autograd as autograd
import torch.optim as optim

import torchvision.transforms as transforms
import torchvision.datasets as dsets
import torchvision.models as models
from logger import Logger
from data_loader import Classify_Dataset
from model import *
from util import *
from path import *

plot = False
DATASET = 'flickr'
MODE = 'train'
LAMDA = 0.5
CV = 1
file_name = argv[1]
CONV_LR = 1e-5
DENSE_LR = 1e-5
lr_decay_rate = 0.95
lr_decay_freq = 30
BATCH_SIZE = 32
test_batch_size = 1
num_workers = 0
EPOCHS = 300
save_fig = True
train_csv_file = train_csv_path.format(DATASET,CV)
val_csv_file = val_csv_path.format(DATASET,CV)
torch.cuda.set_device(1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
weights = torch.FloatTensor([1.21512642,0.99411555,0.22662602,2.77087475,7.61612022,3.09722222,2.40301724,1.8962585]).to(device)


def main():
    base_model = models.squeezenet1_1(pretrained=True)
    #base_model = models.resnet18(pretrained=True)
    #base_model = models.resnet50(pretrained=True)
    #modules = list(base_model.children())[:-1] # delete the last fc layer.
    #base_model = nn.Sequential(*modules)
    model = Classification(base_model).to(device)

    loss_fn = nn.MultiLabelSoftMarginLoss(weight=weights)
    #loss_fn = softCrossEntropy()
    metrics = torch.nn.Sigmoid()

    train_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10, resample=PIL.Image.BILINEAR),
    transforms.ToTensor()])

    val_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()])

    if MODE == 'train':
        if not os.path.exists(ckpt_path+file_name+'/'):
            os.mkdir(ckpt_path+file_name)
        save_path = ckpt_path+file_name+'/'
        logger = Logger(save_path)

        trainset = Classify_Dataset(csv_file=train_csv_file, root_dir=img_path, transform=train_transform)
        valset = Classify_Dataset(csv_file=val_csv_file, root_dir=img_path, transform=val_transform)
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE,
            shuffle=True)
        val_loader = torch.utils.data.DataLoader(valset, batch_size=BATCH_SIZE,
            shuffle=False)

        conv_base_lr = CONV_LR
        dense_lr = DENSE_LR
        optimizer = optim.Adam([{'params': model.features.parameters(), 'lr': conv_base_lr},
                                {'params': model.classifier.parameters(), 'lr': dense_lr}])

        # send hyperparams
        info = ({
            'batch_size': BATCH_SIZE,
            'conv_base_lr': conv_base_lr,
            'dense_lr': dense_lr,
            'lr_decay_rate': lr_decay_rate,
            'lr_decay_freq': lr_decay_freq,
            })
        for tag, value in info.items():
            logger.scalar_summary(tag, value, 0)

        param_num = 0
        for param in model.parameters():
            param_num += int(np.prod(param.shape))
        print('Trainable params: %.2f million' % (param_num / 1e6))
        #loss_fn = softCrossEntropy()

        best_val_acc = float('inf') * -1
        train_losses, val_losses = [], []
        train_acces, val_acces = [], []
        for epoch in range(0, EPOCHS):
            train_loss = 0.0
            train_acc = 0.0
            val_loss = 0.0
            val_acc = 0.0
            start = time.time()
            for batch_idx, data in enumerate(train_loader):
                images = data['image'].to(device)
                labels = data['annotations'].squeeze(2).to(device).float()
                model.train()
                outputs = model(images)
                optimizer.zero_grad()
                output_label = torch.argmax(metrics(outputs),1)
                target_label = torch.argmax(labels,1)
                loss = loss_fn(outputs,labels)
                #print(outputs.shape, labels.shape)
                Acc = np.sum((output_label.cpu() == target_label.cpu()).numpy())
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                train_acc += Acc

                if batch_idx > 0:
                    print('\rCV{} Train Epoch: {}/{} | CE_Loss: {:.6f} | Acc: {:.6f} | [{}/{} ({:.0f}%)] | Time: {}  '.format(
                    CV, epoch+1, EPOCHS, loss, Acc/len(images),
                    batch_idx*BATCH_SIZE, len(train_loader.dataset),
                    100. * batch_idx*BATCH_SIZE / len(train_loader.dataset),
                    timeSince(start, batch_idx*BATCH_SIZE/len(train_loader.dataset))),end='')

            # do validation after each epoch
            for batch_idx, data in enumerate(val_loader):
                images = data['image'].to(device)
                labels = data['annotations'].squeeze(2).to(device).float()
                with torch.no_grad():
                    model.eval()
                    outputs = model(images)
                    outputs = metrics(outputs)
                output_label = torch.argmax(outputs,1)
                target_label = torch.argmax(labels,1)
                optimizer.zero_grad()
                loss = loss_fn(outputs,labels)
                Acc = np.sum((output_label.cpu() == target_label.cpu()).numpy())
                val_loss += loss.item()
                val_acc += Acc

            train_losses.append(train_loss/len(train_loader))
            train_acces.append(train_acc/len(train_loader.dataset))
            val_losses.append(val_loss/len(val_loader))
            val_acces.append(val_acc/len(val_loader.dataset))   
            
            info = {'conv_base_lr': conv_base_lr,
                    'dense_lr': dense_lr,
                    'train loss': train_losses[-1],
                    'train acc': train_acces[-1],
                    'val loss': val_losses[-1],
                    'val acc': val_acces[-1]}

            for tag, value in info.items():
                logger.scalar_summary(tag, value, epoch+1)

            print('\ntrain CE %.4f | train acc %.4f | valid CE %.4f | valid acc %.4f' % (train_losses[-1], train_acces[-1], val_losses[-1], val_acces[-1]))

            # Use early stopping to monitor training
            if val_acces[-1] > best_val_acc:
                best_val_acc = val_acces[-1]
                # save model weights if val loss decreases
                torch.save(model.state_dict(), os.path.join(save_path, 'Acc-%f.pkl' % (best_val_acc)))
                print ('Save Improved Model(Val_acc = %.6f)...' % (best_val_acc))
                # reset stop_count
            if save_fig and (epoch+1) % 20 == 0:
                epochs = range(1, epoch + 2)
                plt.plot(epochs, train_losses, 'b-', label='train CE')
                plt.plot(epochs, val_losses, 'g-', label='val CE')
                plt.plot(epochs, train_acces, 'r-', label='train acc')
                plt.plot(epochs, val_acces, 'y', label='val acc')
                plt.title('CE loss')
                plt.legend()
                plt.savefig(save_path+'loss.png')
                plt.close()

if __name__ == '__main__':
    main()

