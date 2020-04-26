import pandas as pd
import numpy as np
from PIL import Image
import sys
import matplotlib.pyplot as plt
import os
import random
from sklearn.utils.class_weight import compute_class_weight



dataset = 'flickr'
filename = '/home/b04020/2018_autumn/CongitveComputing/Affective Image/data/'+dataset+'/ground_truth.txt'
root_dir = '/home/b04020/2018_autumn/CongitveComputing/Affective Image/data/'+dataset+'/'
savepath = '/home/b04020/2018_autumn/CongitveComputing/Affective Image/train_val/'+dataset
label = ['Amusement','Awe','Contentment','Excitement','Anger','Disgust','Fear','Sadness']

annotation = pd.read_csv(filename,sep='\s+').values
print(annotation.shape)
'''
for i, name in enumerate(annotation[:,0]):
	img_name = os.path.join(root_dir, name)
	image = np.array(Image.open(img_name))
	if len(image.shape) < 3 or image.shape[2] != 3 :
		print('\n'+str(img_name))
	else:
		if i == 0:
			new_annotation = [annotation[i]]
		else:
			new_annotation += [annotation[i]]
	print('\r{} {} {}'.format(str(i),str(image.shape),str(np.array(new_annotation).shape)),end='',)
'''
#annotation = np.array(new_annotation)
#np.random.shuffle(annotation)
img_name = annotation[:,0]
rating = annotation[:,1:]

'''
cv = int(annotation.shape[0]*0.2)
for i in range(5):
	val = annotation[cv*i:cv*(i+1)]
	train = np.array(annotation[:cv*i].tolist()+annotation[cv*(i+1):].tolist())
	file = open('{}/train{}.csv'.format(savepath,i+1),'w')
	for j in train:
		for k in j:
			file.write(str(k)+' ')
		file.write('\n')
	file.close()
	file = open('{}/val{}.csv'.format(savepath,i+1),'w')
	for j in val:
		for k in j:
			file.write(str(k)+' ')
		file.write('\n')
	file.close()
'''
value = []
collect = np.zeros(8)
for v in rating:
	i_d = np.argmax(v)
	collect[i_d] += 1
for i, num in enumerate(collect):
	print('{}: {}%({})'.format(label[i], np.around(num/(np.sum(collect))*100,2) ,int(num)))

y = np.argmax(rating,axis=1)
weights = compute_class_weight('balanced', np.unique(y), y)
print(weights)
'''
for i, name in enumerate(img_name):
	path = os.path.join(root_dir, str(name))
	print(path)
	image = Image.open(path)
	print(rating[i])
	plt.imshow(image)
	plt.show()
'''


