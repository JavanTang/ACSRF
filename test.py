import math
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
# from pytorch_msssim import ssim
from torch.utils.data import DataLoader
from collections import OrderedDict

from utils import AverageMeter, write_img, chw_to_hwc, pad_img
from datasets.loader import PairLoader
from models import *
from metrics import ssim, psnr, epe
# from metrics_2 import ssim, psnr

# torch.cuda.set_per_process_memory_fraction(0.1)


# python test.py --data_dir=/home/tangzhifeng/codes/Dataset --test_set=I-HAZY-TEST  --model=cxnet14_b --exp=i-hazy
parser = argparse.ArgumentParser()
parser.add_argument('--model', default='cxnet14_b', type=str, help='model name')
parser.add_argument('--num_workers', default=16, type=int, help='number of workers')
parser.add_argument('--data_dir', default='../Dataset', type=str, help='path to dataset')
parser.add_argument('--save_dir', default='./saved_models/', type=str, help='path to models saving')
parser.add_argument('--result_dir', default='./results/', type=str, help='path to results saving')
parser.add_argument('--test_set', default='I-HAZY-TEST', type=str, help='test dataset name')
parser.add_argument('--exp', default='i-hazy', type=str, help='experiment setting')
args = parser.parse_args()


def get_current_time():
	import datetime

	# Get the current time
	current_time = datetime.datetime.now()

	# Format the time as a string in the desired format
	formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S")

	# Output the time
	return formatted_time


def single(save_dir):
	state_dict = torch.load(save_dir)['state_dict']
	new_state_dict = OrderedDict()

	for k, v in state_dict.items():
		name = k[7:]
		new_state_dict[name] = v

	return new_state_dict


def to_psnr(dehaze, gt):
    mse = F.mse_loss(dehaze, gt, reduction='none')
    mse_split = torch.split(mse, 1, dim=0)
    mse_list = [torch.mean(torch.squeeze(mse_split[ind])).item() for ind in range(len(mse_split))]

    intensity_max = 1.0
    psnr_list = [10.0 * math.log10(intensity_max / mse) for mse in mse_list]
    return psnr_list

def test(test_loader, network, result_dir, img_name='imgs', result_name='results.csv', cycle_dehazing_size = 1):
	PSNR = AverageMeter()
	SSIM = AverageMeter()
	EPE = AverageMeter()

	torch.cuda.empty_cache()
	network.cuda()
	network.eval()
	

	os.makedirs(os.path.join(result_dir, img_name), exist_ok=True)
	f_result = open(os.path.join(result_dir, result_name), 'w')

	_size = cycle_dehazing_size
	for idx, batch in enumerate(test_loader):
		
		_size = cycle_dehazing_size
		input = batch['source'].cuda()
		target = batch['target'].cuda()

		print('input.shape = ', input.shape)
		print('target.shape = ', target.shape)		

		filename = batch['filename'][0]

		with torch.no_grad():
			H, W = input.shape[2:]
			input = pad_img(input, (512,512))
			
			# cycle dehazing
			cycel_input = input

			cycel_input = pad_img(cycel_input, 512)

			while _size >= 1:
				if network.vis is not None:
					cycel_input, attention_map = network(cycel_input)
				else:
					cycel_input = network(cycel_input)
				_size -= 1
			
			output = cycel_input


			output = output * 0.5 + 0.5
			target = target * 0.5 + 0.5
			
			target = pad_img(target, 512)
			output = pad_img(output, 512)

			# print('output.shape = ', output.shape)
			# print('target.shape = ', target.shape)
			_ssim = ssim(output, target).item()
			_psnr = psnr(output, target)
			_epe = epe(output, target)
				

			PSNR.update(_psnr)
			SSIM.update(_ssim)
			EPE.update(_epe)

			print('Test: [{0}]\t'
				'PSNR: {psnr.val:.02f} ({psnr.avg:.02f})\t'
				'SSIM: {ssim.val:.03f} ({ssim.avg:.03f}), EPE: {epe.val:.03f} ({epe.avg:.03f})'.format(
				idx, psnr=PSNR, ssim=SSIM, epe=EPE))

			f_result.write('%s,%.02f,%.03f\n'%(filename, _psnr, _ssim))

			# print('min_input = %.03f, max_input = %.03f'%(input.min(), input.max()))
			# print('min_output = %.03f, max_output = %.03f'%(output.min(), output.max()))
			# print('min_target = %.03f, max_target = %.03f'%(target.min(), target.max()))

			# input = input * 0.5 + 0.5

			input = chw_to_hwc(input[0].cpu().numpy())
			# input = cv2.resize(input, (W, H))
			input = cv2.cvtColor(input, cv2.COLOR_BGR2RGB)

			# output = output * 0.5 + 0.5
			output = chw_to_hwc(output[0].cpu().numpy())
			# output = cv2.resize(output, (W, H))
			output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
			

			# target = target * 0.5 + 0.5
			target = chw_to_hwc(target[0].cpu().numpy())
			# target = cv2.resize(target, (W, H))
			target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)

			write_img(os.path.join(result_dir, img_name, filename[:-4]+'_input.png'), input)
			write_img(os.path.join(result_dir, img_name, filename[:-4]+'_predict.png'), output)
			write_img(os.path.join(result_dir, img_name, filename[:-4]+'_gt.png'), target)

			# Cleanup
			# input, output, target, cycel_input, attention_map = None, None, None, None, None
			# _psnr, _ssim, _epe, H, W = None, None, None, None, None
			del input, output, target, cycel_input
			_psnr, _ssim, _epe, H, W = None, None, None, None, None

			torch.cuda.empty_cache()
			import gc
			gc.collect()


	f_result.close()

	os.rename(os.path.join(result_dir, result_name), 
			  os.path.join(result_dir, result_name.replace(".csv", "") + ' -> %.03f | %.04f.csv'%(PSNR.avg, SSIM.avg)))


def main():
	network = eval(args.model)()
	network.cuda()
	saved_model_dir = os.path.join(args.save_dir, args.exp, args.model+'.pth')

	if os.path.exists(saved_model_dir):
		print('==> Start testing, current model name: ' + args.model)
		network.load_state_dict(single(saved_model_dir))
	else:
		print('==> No existing best trained model!')
		exit(0)
	
	# 如果是attention skip attention的话，就打印一下attention map
	try:
		if network.vis is not None:
			network.vis = True
	except:
		network.vis = None

	dataset_dir = os.path.join(args.data_dir, args.test_set)
	test_dataset = PairLoader(dataset_dir, 'test')
	test_loader = DataLoader(test_dataset,
							 batch_size=1,
							 num_workers=args.num_workers,
							 pin_memory=True)

	result_dir = os.path.join(args.result_dir, args.test_set, args.exp, args.model, get_current_time())

	test(test_loader, network, result_dir, img_name='img_best_model', result_name='results_best_model.csv',cycle_dehazing_size=1)


	network = eval(args.model)()
	network.cuda()
	saved_model_dir = os.path.join(args.save_dir, args.exp, args.model+'_last.pth')

	try:
		if network.vis is not None:
			network.vis = True
	except:
		network.vis = None

	if os.path.exists(saved_model_dir):
		print('==> Start testing, current model name: ' + args.model)
		network.load_state_dict(single(saved_model_dir))
	else:
		print('==> No existing last trained model!')
		exit(0)
	test(test_loader, network, result_dir, img_name='img_last_model', result_name='results_last_model.csv',cycle_dehazing_size=1)


if __name__ == '__main__':
	main()
