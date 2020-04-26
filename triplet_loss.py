import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class TripletLoss(nn.Module):
	"""
	Triplet loss
	Takes embeddings of an anchor sample, a positive sample and a negative sample
	"""
	def __init__(self, margin):
		super(TripletLoss, self).__init__()
		self.margin = margin

	def forward(self, anchor, positive, negative, size_average=True):
		distance_positive = (anchor - positive).pow(2).sum(1).sqrt()  # .pow(.5)
		distance_negative = (anchor - negative).pow(2).sum(1).sqrt()  # .pow(.5)
		#print('\npos: {}'.format(distance_positive))
		#print('neg: {}'.format(distance_negative))
		losses = F.relu(distance_positive.pow(2) - distance_negative.pow(2) + self.margin)
		return losses.mean() if size_average else losses.sum()

	def get_distance(self, output1, output2):
		return np.sqrt(np.sum(np.square(output2 - output1)))

class SentimentLoss(nn.Module):
	def __init__(self, margin):
		super(SentimentLoss, self).__init__()
		self.margin = margin

	def forward(self, anchor, positive, relative, negative, size_average=True):

		distance_positive = (anchor - positive).pow(2).sum(1).sqrt()  # .pow(.5)
		distance_relative = (anchor - relative).pow(2).sum(1).sqrt() 
		distance_negative = (anchor - negative).pow(2).sum(1).sqrt()  # .pow(.5)
		losses = F.relu(distance_positive.pow(2) - distance_relative.pow(2) + self.margin) + F.relu(distance_relative.pow(2) - distance_negative.pow(2) + self.margin)
		return losses.mean() if size_average else losses.sum()

	def get_distance(self, output1, output2):
		return np.sqrt(np.sum(np.square(output2 - output1)))