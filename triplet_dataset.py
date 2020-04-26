import numpy as np
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
import torch
import os

class Concate_Dataset(Dataset):
    def __init__(self, dataset1, dataset2):
        self.Classify_Dataset = dataset1
        self.TripletDataset = dataset2
    def __len__(self):
        return self.Classify_Dataset.__len__()
    def __getitem__(self, idx):
        sample = self.Classify_Dataset.__getitem__(idx)
        data, target = self.TripletDataset.__getitem__(idx)
        return sample, data, target


class Classify_Dataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.annotations = pd.read_csv(csv_file,sep='\s+')
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, str(self.annotations.iloc[idx, 0]))
        image = Image.open(img_name)
        annotations = self.annotations.iloc[idx, 1:].values
        annotations = annotations/11
        annotations = annotations.astype('float').reshape(-1, 1)
        sample = {'img_id': img_name, 'image': image, 'annotations': annotations}
        if self.transform:
            sample['image'] = self.transform(sample['image'])
            if sample['image'].shape[0] != 3:
                print(img_name)
                print(sample['image'].shape)
        return sample


class TripletDataset(Dataset):
    def __init__(self, csv_file, root_dir, task, transform=None):
        self.annotations = pd.read_csv(csv_file,sep='\s+')
        self.data = self.annotations.iloc[:,0].values
        self.label = np.argmax(self.annotations.iloc[:,1:].values,axis=1)
        self.transform = transform
        self.root_dir = root_dir
        self.task = task
        self.label_set = set(self.label)
        self.label_to_indices = {lab: np.where(self.label == lab)[0] for lab in self.label_set}

        if self.task=='test':
            random_state = np.random.RandomState(29)
            triplet = [[i,
                        random_state.choice(self.label_to_indices[self.label[i]]),
                        random_state.choice(self.label_to_indices[np.random.choice(list(self.label_set - set([self.label[i]])))])]
                        for i in range(len(self.data))]
            self.test_triplet = triplet

    def __getitem__(self,index):
        if self.task=='train':
            sentence1, label1 = self.data[index], self.label[index]
            positive_index = index
            while positive_index == index:
                positive_index = np.random.choice(self.label_to_indices[label1])
            negative_label = np.random.choice(list(self.label_set - set([label1])))
            negative_index = np.random.choice(self.label_to_indices[negative_label])
            sentence2 = self.data[positive_index]
            sentence3 = self.data[negative_index]
        elif self.task=='test':
            sentence1 = self.data[self.test_triplet[index][0]]
            sentence2 = self.data[self.test_triplet[index][1]]
            sentence3 = self.data[self.test_triplet[index][2]]
        sentence1 = Image.open(os.path.join(self.root_dir, str(sentence1)))
        sentence2 = Image.open(os.path.join(self.root_dir, str(sentence2)))
        sentence3 = Image.open(os.path.join(self.root_dir, str(sentence3)))
        if self.transform:
            sentence1 = self.transform(sentence1)
            sentence2 = self.transform(sentence2)
            sentence3 = self.transform(sentence3)
        return (sentence1,sentence2,sentence3),[]
    def __len__(self):
       return len(self.annotations)

class SentimentDataset(Dataset):
    def __init__(self, csv_file, root_dir, task, transform=None):
        self.sentilabel_group = [[0,1,2,3],[4,5,6,7]]
        self.sentilabel_pos = {0:0, 1:0, 2:0, 3:0, 4:1, 5:1, 6:1, 7:1}
        self.annotations = pd.read_csv(csv_file,sep='\s+')
        self.data = self.annotations.iloc[:,0].values
        self.label = np.argmax(self.annotations.iloc[:,1:].values,axis=1)
        self.transform = transform
        self.root_dir = root_dir
        self.task = task
        self.label_set = set(self.label)
        self.label_to_indices = {lab: np.where(self.label == lab)[0] for lab in self.label_set} #idx can choose
        if self.task=='test':
            random_state = np.random.RandomState(29)
            triplet = [[i,
                        random_state.choice(self.label_to_indices[self.label[i]]),
                        random_state.choice(self.label_to_indices[np.random.choice(list(set(self.sentilabel_group[self.sentilabel_pos[self.label[i]]])-set([self.label[i]])))]),
                        random_state.choice(self.label_to_indices[np.random.choice(list(self.label_set - set(self.sentilabel_group[self.sentilabel_pos[self.label[i]]])))])]
                        for i in range(len(self.data))]
            self.test_triplet = triplet

    def __getitem__(self,index):
        if self.task=='train':
            sentence_anchor, label_anchor = self.data[index], self.label[index]
            positive_index = index
            while positive_index == index:
                positive_index = np.random.choice(self.label_to_indices[label_anchor])
            relative_label = np.random.choice(list(set(self.sentilabel_group[self.sentilabel_pos[label_anchor]])-set([label_anchor])))
            relative_index = np.random.choice(self.label_to_indices[relative_label])
            negative_label = np.random.choice(list(self.label_set - set(self.sentilabel_group[self.sentilabel_pos[label_anchor]])))
            negative_index = np.random.choice(self.label_to_indices[negative_label])
            sentence_positive = self.data[positive_index]
            sentence_relative = self.data[relative_index]
            sentence_negative = self.data[negative_index]
        elif self.task=='test':
            sentence_anchor = self.data[self.test_triplet[index][0]]
            sentence_positive = self.data[self.test_triplet[index][1]]
            sentence_relative = self.data[self.test_triplet[index][2]]
            sentence_negative = self.data[self.test_triplet[index][3]]  
        sentence_anchor = Image.open(os.path.join(self.root_dir, str(sentence_anchor)))
        sentence_positive = Image.open(os.path.join(self.root_dir, str(sentence_positive)))
        sentence_relative = Image.open(os.path.join(self.root_dir, str(sentence_relative)))
        sentence_negative = Image.open(os.path.join(self.root_dir, str(sentence_negative)))
        if self.transform:
            sentence_anchor = self.transform(sentence_anchor)
            sentence_positive = self.transform(sentence_positive)
            sentence_relative = self.transform(sentence_relative)
            sentence_negative = self.transform(sentence_negative)
        return (sentence_anchor,sentence_positive,sentence_relative,sentence_negative),[]
    def __len__(self):
        return len(self.annotations)