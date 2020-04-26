import torch
import torch.nn as nn
import torch.nn.functional as F

class Classification(nn.Module):
    def __init__(self, base_model,num_classes=8):
        super(Classification, self).__init__()
        #self.features = base_model
        self.features = base_model.features
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.75),
            nn.Linear(in_features=86528, out_features=num_classes))
    def forward(self, x, embedding=False):
        emb = self.features(x)
        emb = emb.view(emb.size(0), -1)
        out = self.classifier(emb)
        if embedding:
            return out, emb
        return out