import os
import scipy.io
from random import shuffle

images_dir = '/home/kh/Miscellaneous/vgg_flowers102/data/jpg'
labels_file = '/home/kh/Miscellaneous/vgg_flowers102/data/imagelabels.mat'

train_file = '/home/kh/train.txt'
val_file = '/home/kh/val.txt'

labels_list = scipy.io.loadmat(labels_file)['labels'][0]

pair_list = []

image_suffix = "image_00000"
index = 1
for label in labels_list:
	image_name = image_suffix
	image_name = image_name[:-len(str(index))] + str(index) + ".jpg" 
	print("Adding {} to the dataset list.".format(image_name))
	image_path = os.path.join(images_dir, image_name)
	pair = [image_path, str(label)]
	pair_list.append(pair)
	index += 1

shuffle(pair_list)

#Split the list into 80/20
train_list = pair_list[:int(0.8*len(pair_list))]
val_list = pair_list[int(0.8*len(pair_list)):]

with open(train_file, 'w') as f:
	for pair in train_list:
		image_path = pair[0]
		label = pair[1]
		write_string = '{} {}\n'.format(image_path, label)
		f.write(write_string)

with open(val_file, 'w') as f:
	for pair in val_list:
		image_path = pair[0]
		label = pair[1]
		write_string = '{} {}\n'.format(image_path, label)
		f.write(write_string)