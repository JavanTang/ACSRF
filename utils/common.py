import numpy as np
import cv2
import torch.nn.functional as F
import torchvision.transforms.functional as fn
from PIL import Image


class AverageMeter(object):
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


class ListAverageMeter(object):
	"""Computes and stores the average and current values of a list"""
	def __init__(self):
		self.len = 10000  # set up the maximum length
		self.reset()

	def reset(self):
		self.val = [0] * self.len
		self.avg = [0] * self.len
		self.sum = [0] * self.len
		self.count = 0

	def set_len(self, n):
		self.len = n
		self.reset()

	def update(self, vals, n=1):
		assert len(vals) == self.len, 'length of vals not equal to self.len'
		self.val = vals
		for i in range(self.len):
			self.sum[i] += self.val[i] * n
		self.count += n
		for i in range(self.len):
			self.avg[i] = self.sum[i] / self.count
			

def read_img(filename, to_float=True):
	img = cv2.imread(filename)
	if to_float: img = img.astype('float32') / 255.0
	return img[:, :, ::-1]


def write_img(filename, img, to_uint=True):
img = img * 255 + 0.5
	img = np.clip(img, 0, 255)
	img = Image.fromarray(np.uint8(img))

	img.save(filename)


def hwc_to_chw(img):
	return np.transpose(img, axes=[2, 0, 1]).copy()


def chw_to_hwc(img):
	return np.transpose(img, axes=[1, 2, 0]).copy()


def pad_img(x, patch_size):
	# 如果patch是一个值，就把它变成一个tuple
	if isinstance(patch_size, int):
		patch_size = (patch_size, patch_size)

	return fn.resize(x,patch_size,interpolation=2)
