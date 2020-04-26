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
#CV = 1
#file_name = argv[1]
CONV_LR = 1e-5
DENSE_LR = 1e-5
lr_decay_rate = 0.95
lr_decay_freq = 30
BATCH_SIZE = 32
test_batch_size = 1
num_workers = 0
EPOCHS = 300
save_fig = True
#train_csv_file = train_csv_path.format(DATASET,CV)
#val_csv_file = val_csv_path.format(DATASET,CV)
torch.cuda.set_device(1)


def main():
    for CV in range(2,6):
        train_csv_file = train_csv_path.format(DATASET,CV)
        val_csv_file = val_csv_path.format(DATASET,CV)
        file_name = 'f_cv{}_squeeze_emd'.format(CV)
        #file_name = argv[1]
        train(CV, train_csv_file,val_csv_file, file_name)


def train(CV, train_csv_file, val_csv_file, file_name):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_model = models.squeezenet1_1(pretrained=True)
    #base_model = models.resnet18(pretrained=True)
    #base_model = models.resnet50(pretrained=True)
    #modules = list(base_model.children())[:-1] # delete the last fc layer.
    #base_model = nn.Sequential(*modules)
    model = Classification(base_model).to(device)

    #loss_fn = nn.MultiLabelSoftMarginLoss()
    #loss_fn = softCrossEntropy()
    #metrics = torch.nn.Sigmoid()
    loss_fn = nn.CrossEntropyLoss()

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
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE,shuffle=True)
        val_loader = torch.utils.data.DataLoader(valset, batch_size=BATCH_SIZE,shuffle=False)

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
        Distance = torch.zeros([8,8])
        u = -0.5
        LAMDA = 0.25
        for epoch in range(0, EPOCHS):
            train_loss = 0.0
            train_acc = 0.0
            val_loss = 0.0
            val_acc = 0.0
            class_vector = {}
            start = time.time()
            for batch_idx, data in enumerate(train_loader):
                images = data['image'].to(device)
                labels = data['annotations'].to(device).float()
                model.train()
                optimizer.zero_grad()
                outputs, embedding = model(images, embedding=True)
                embedding = embedding.cpu()
                output_label = torch.argmax(outputs,1)
                target_label = torch.argmax(labels,1).squeeze(1)

                if epoch > 3:
                    EMD_loss = torch.sum(torch.mul(F.softmax(outputs,dim=1),(Distance[target_label.cpu().numpy()] + u).to(device)))
                    loss = (1 - LAMDA) * loss_fn(outputs,target_label) + LAMDA * EMD_loss
                else:
                    loss = loss_fn(outputs,target_label)

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

                for n, label in enumerate(target_label.cpu().numpy()):
                    if label not in class_vector:
                        class_vector[label] = [embedding[n]]
                    else:
                        class_vector[label].append(embedding[n])

            label_mean = []
            for label in sorted(list(class_vector.keys())):
                label_mean.append(torch.mean(torch.stack(class_vector[label]),dim=0))
            Distance_temp = torch.zeros_like(Distance)
            for x in range(8):
                for y in range(8):
                    Distance[x][y] = torch.sqrt(torch.sum(torch.pow(label_mean[x] - label_mean[y], 2)))
            for x in range(8):
                for y in range(8):
                    count = 0
                    for i in range(8):
                        if Distance[x][i] < Distance[x][y]:
                            count += 1
                    Distance_temp[x][y] = count/8
            Distance = (Distance_temp+torch.transpose(Distance_temp,0,1))/2
            print('\n')
            print(Distance)
            np.save(os.path.join(save_path, 'Distance.npy'),Distance.numpy())

            # do validation after each epoch
            for batch_idx, data in enumerate(val_loader):
                images = data['image'].to(device)
                labels = data['annotations'].to(device).float()
                with torch.no_grad():
                    model.eval()
                    outputs = model(images)
                output_label = torch.argmax(outputs,1)
                target_label = torch.argmax(labels,1).squeeze(1)
                loss = loss_fn(outputs,target_label)
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

