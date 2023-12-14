import os
import random
import numpy as np
import cv2

from random import randint
from torch.utils.data import Dataset
from utils import hwc_to_chw, read_img


def preprocess_image_cv2_rancrop_flip(img_A, img_B, RESHAPE=(512,512)):
	# 数据增强的地方!!

	img_A_flip = cv2.flip(img_A, 1)
	img_B_flip = cv2.flip(img_B, 1)

	h, w, _ = img_A.shape

	min_wh = np.amin([h, w])

	# crop_sizes = [1600, 1800, 2000, 2200, 2400]
	crop_sizes = [int(min_wh*0.4), int(min_wh*0.5), int(min_wh*0.6), int(min_wh*0.7), int(min_wh*0.8)]


	images_A = []
	images_B = []

	for crop_size in crop_sizes:

		x1, y1 = randint(1, w-crop_size-1), randint(1, h-crop_size-1)

		# Original

		cropA = img_A[y1:y1+crop_size,x1:x1+int(crop_size/h*w)]
		cropA = cv2.resize(cropA, (RESHAPE))
		cropA = np.array(cropA)

		# crop_t = t[y1:y1+crop_size,x1:x1+int(crop_size/h*w)]
		# crop_t = preprocess_depth_img(crop_t)

		# cropA = np.concatenate((cropA, crop_t), axis=2)

		cropB = img_B[y1:y1+crop_size,x1:x1+int(crop_size/h*w)]
		cropB = cv2.resize(cropB, (RESHAPE))
		cropB = np.array(cropB)

		images_A.append(cropA)
		images_B.append(cropB)


		# Horizontal Flip
		cropA = img_A_flip[y1:y1+crop_size,x1:x1+int(crop_size/h*w)]
		cropA = cv2.resize(cropA, (RESHAPE))
		cropA = np.array(cropA)

		# crop_t = t_flip[y1:y1+crop_size,x1:x1+int(crop_size/h*w)]
		# crop_t = preprocess_depth_img(crop_t)

		# cropA = np.concatenate((cropA, crop_t), axis=2)

		cropB = img_B_flip[y1:y1+crop_size,x1:x1+int(crop_size/h*w)]
		cropB = cv2.resize(cropB, (RESHAPE))
		cropB = np.array(cropB)

		images_A.append(cropA)
		images_B.append(cropB)


	# Original

	img_A = cv2.resize(img_A, (RESHAPE))
	img_A = np.array(img_A)
	# img_A = (img_A - 127.5) / 127.5

	# t = preprocess_depth_img(t)

	# img_A = np.concatenate((img_A, t), axis=2)

	img_B = cv2.resize(img_B, (RESHAPE))
	img_B = np.array(img_B)
	# img_B = (img_B - 127.5) / 127.5

	images_A.append(img_A)
	images_B.append(img_B)


	# Horizontal Flip

	img_A = cv2.resize(img_A_flip, (RESHAPE))
	img_A = np.array(img_A)
	# img_A = (img_A - 127.5) / 127.5

	# t = preprocess_depth_img(t_flip)

	# img_A = np.concatenate((img_A, t), axis=2)

	img_B = cv2.resize(img_B_flip, (RESHAPE))
	img_B = np.array(img_B)
	# img_B = (img_B - 127.5) / 127.5

	images_A.append(img_A)
	images_B.append(img_B)

	return images_A, images_B

def augment(imgs=[], size=256, edge_decay=0., data_augment=True):
	H, W, _ = imgs[0].shape
	Hc, Wc = [size, size]

	# simple re-weight for the edge
	if random.random() < Hc / H * edge_decay:
		Hs = 0 if random.randint(0, 1) == 0 else H - Hc
	else:
		Hs = random.randint(0, H-Hc)

	if random.random() < Wc / W * edge_decay:
		Ws = 0 if random.randint(0, 1) == 0 else W - Wc
	else:
		Ws = random.randint(0, W-Wc)

	for i in range(len(imgs)):
		imgs[i] = imgs[i][Hs:(Hs+Hc), Ws:(Ws+Wc), :]

	if data_augment:
		# horizontal flip
		if random.randint(0, 1) == 1:
			for i in range(len(imgs)):
				imgs[i] = np.flip(imgs[i], axis=1)

		# bad data augmentations for outdoor dehazing
		rot_deg = random.randint(0, 3)
		for i in range(len(imgs)):
			imgs[i] = np.rot90(imgs[i], rot_deg, (0, 1))
			
	return imgs


def align(imgs=[], size=256):
	H, W, _ = imgs[0].shape
	Hc, Wc = [size, size]

	Hs = (H - Hc) // 2
	Ws = (W - Wc) // 2
	for i in range(len(imgs)):
		imgs[i] = imgs[i][Hs:(Hs+Hc), Ws:(Ws+Wc), :]

	return imgs

class PairLoader(Dataset):
	def __init__(self, root_dir, mode, size=256, edge_decay=0, data_augment=True, cache_memory=False):
		assert mode in ['train', 'valid', 'test']

		self.mode = mode
		self.size = size
		self.edge_decay = edge_decay
		self.data_augment = data_augment

		# 读取图像名称
		self.root_dir = root_dir
		self.GT_dir = sorted(os.listdir(os.path.join(self.root_dir, 'GT')))
		self.IN_dir = sorted(os.listdir(os.path.join(self.root_dir, 'IN')))

		# print(self.IN_dir)
		self.img_names = [i for i in self.GT_dir if i.lower().endswith('.jpg') or i.lower().endswith('.png')]
		self.hazy_img_names = [i for i in self.IN_dir if i.lower().endswith('.jpg') or i.lower().endswith('.png')]

		# 将图片读取并且做图像增强
		self.data_sourses = []
		self.data_targets = []
		self.paths = []

		if mode == 'train':
			# train 这里需要做 剪切+水平翻转
			for i in range(len(self.img_names)):
				path = self.img_names[i]
				img = cv2.imread(os.path.join(self.root_dir, 'GT', self.img_names[i])).astype('float32') 
				hazy_img = cv2.imread(os.path.join(self.root_dir, 'IN', self.hazy_img_names[i])).astype('float32')
				
				preprocess_img_A, preprocess_img_B = preprocess_image_cv2_rancrop_flip(hazy_img, img, (self.size, self.size))
				for j in range(len(preprocess_img_A)):
					self.data_sourses.append(preprocess_img_A[j])
					self.data_targets.append(preprocess_img_B[j])
					self.paths.append(path)
		else:
			# test 这里需要做resize
			for i in range(len(self.img_names)):
				path = self.img_names[i]
				img = cv2.imread(os.path.join(self.root_dir, 'GT', self.img_names[i])).astype('float32')
				hazy_img = cv2.imread(os.path.join(self.root_dir, 'IN', self.hazy_img_names[i])).astype('float32')
				
				self.data_sourses.append(hazy_img)
				self.data_targets.append(img)
				self.paths.append(path)

		# 总数
		self.img_num = len(self.data_sourses)

		self.cache_memory = cache_memory and mode == 'train'
		self.source_files = {}
		self.target_files = {}

	def __len__(self):
		return self.img_num

	def __getitem__(self, idx):
		cv2.setNumThreads(0)
		cv2.ocl.setUseOpenCL(False)

		source_img = self.data_sourses[idx]
		target_img = self.data_targets[idx]
		path = self.paths[idx]

		# [0, 1] to [-1, 1]
		source_img = source_img.astype('float32') / 255.0 * 2 - 1
		target_img = target_img.astype('float32') / 255.0 * 2 - 1
		
		# data augmentation
		# if self.mode == 'train':
		# 	[source_img, target_img] = augment([source_img, target_img], self.size, self.edge_decay, self.data_augment)

		# if self.mode == 'valid':
		# 	[source_img, target_img] = align([source_img, target_img], self.size)

		return {'source': hwc_to_chw(source_img), 'target': hwc_to_chw(target_img), 'filename': path}


# class PairLoader(Dataset):
# 	def __init__(self, root_dir, mode, size=256, edge_decay=0, data_augment=True, cache_memory=False):
# 		assert mode in ['train', 'valid', 'test']

# 		self.mode = mode
# 		self.size = size
# 		self.edge_decay = edge_decay
# 		self.data_augment = data_augment

# 		self.root_dir = root_dir
# 		self.img_names = sorted(os.listdir(os.path.join(self.root_dir, 'GT')))
# 		self.hazy_img_names = sorted(os.listdir(os.path.join(self.root_dir, 'IN')))

# 		self.img_num = len(self.img_names)

# 		self.cache_memory = cache_memory and mode == 'train'
# 		self.source_files = {}
# 		self.target_files = {}

# 	def __len__(self):
# 		return self.img_num

# 	def __getitem__(self, idx):
# 		cv2.setNumThreads(0)
# 		cv2.ocl.setUseOpenCL(False)

# 		# select a image pair
# 		img_name = self.img_names[idx]
# 		hazy_img_name = self.hazy_img_names[idx]

# 		# read images
# 		if img_name not in self.source_files:
# 			source_img = read_img(os.path.join(self.root_dir, 'IN', hazy_img_name), to_float=False)
# 			target_img = read_img(os.path.join(self.root_dir, 'GT', img_name), to_float=False)

# 			# cache in memory if specific (uint8 to save memory)
# 			if self.cache_memory:
# 				self.source_files[hazy_img_name] = source_img
# 				self.target_files[img_name] = target_img
# 		else:
# 			# load cached images
# 			source_img = self.source_files[hazy_img_name]
# 			target_img = self.target_files[img_name]

# 		# [0, 1] to [-1, 1]
# 		source_img = source_img.astype('float32') / 255.0 * 2 - 1
# 		target_img = target_img.astype('float32') / 255.0 * 2 - 1
		
# 		# data augmentation
# 		if self.mode == 'train':
# 			[source_img, target_img] = augment([source_img, target_img], self.size, self.edge_decay, self.data_augment)

# 		if self.mode == 'valid':
# 			[source_img, target_img] = align([source_img, target_img], self.size)

# 		return {'source': hwc_to_chw(source_img), 'target': hwc_to_chw(target_img), 'filename': img_name}


class SingleLoader(Dataset):
	def __init__(self, root_dir):
		self.root_dir = root_dir
		self.img_names = sorted(os.listdir(self.root_dir))
		self.img_num = len(self.img_names)

	def __len__(self):
		return self.img_num

	def __getitem__(self, idx):
		cv2.setNumThreads(0)
		cv2.ocl.setUseOpenCL(False)

		img_name = self.img_names[idx]
		img = read_img(os.path.join(self.root_dir, img_name)) * 2 - 1

		return {'img': hwc_to_chw(img), 'filename': img_name}
