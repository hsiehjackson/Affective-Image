import torch
import torch.nn as nn
import torch.nn.functional as F

class EmbeddingNET(nn.Module):
    def __init__(self, base_model):
        super(EmbeddingNET, self).__init__()
        #self.features = base_model
        self.features = base_model.features
    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        return out

class ClassifierNET(nn.Module):
    def __init__(self, embedding_net, num_classes=8):
        super(ClassifierNET, self).__init__()
        self.embedding_net = embedding_net
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.75),
            nn.Linear(in_features=86528, out_features=num_classes))
    def forward(self, x):
        out = self.embedding_net(x)
        out = self.classifier(out)
        return out

class TripletNet(nn.Module):
    def __init__(self, embedding_net):
        super(TripletNet, self).__init__()
        self.embedding_net = embedding_net
        #self.embedding_net.embedding_layer.weight.requires_grad = False
    def forward(self, x1, x2, x3):
        output1 = self.embedding_net(x1)
        output2 = self.embedding_net(x2)
        output3 = self.embedding_net(x3)
        return output1, output2, output3
    def get_embedding(self, x):
        return self.embedding_net(x)

class SentimentNet(nn.Module):
    def __init__(self, embedding_net):
        super(SentimentNet, self).__init__()
        self.embedding_net = embedding_net
        #self.embedding_net.embedding_layer.weight.requires_grad = False
    def forward(self, x1, x2, x3, x4):
        output1 = self.embedding_net(x1)
        output2 = self.embedding_net(x2)
        output3 = self.embedding_net(x3)
        output4 = self.embedding_net(x4)
        return output1, output2, output3, output4
    def get_embedding(self, x):
        return self.embedding_net(x)