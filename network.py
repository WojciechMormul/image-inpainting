import numpy as np
import cv2
import torch
from torch.autograd import Variable
from roi_align.roi_align import CropAndResize

class Generator(torch.nn.Module):
	
	def __init__(self):
			
		super(Generator, self).__init__()

		self.generator = torch.nn.Sequential(

			torch.nn.Conv2d(in_channels=4, out_channels=64, kernel_size=5, stride=1, padding=2),
			torch.nn.BatchNorm2d(64),
			torch.nn.ReLU(),
			torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1),
			torch.nn.BatchNorm2d(128),
			torch.nn.ReLU(),
			torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
			torch.nn.BatchNorm2d(128),
			torch.nn.ReLU(),
			torch.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1),
			torch.nn.BatchNorm2d(256),
			torch.nn.ReLU(),
			torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
			torch.nn.BatchNorm2d(256),
			torch.nn.ReLU(),
			torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
			torch.nn.BatchNorm2d(256),
			torch.nn.ReLU(),
			torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=2, dilation=2),
			torch.nn.BatchNorm2d(256),
			torch.nn.ReLU(),
			torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=4, dilation=4),
			torch.nn.BatchNorm2d(256),
			torch.nn.ReLU(),
			torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=8, dilation=8),
			torch.nn.BatchNorm2d(256),
			torch.nn.ReLU(),
			torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=16, dilation=16),
			torch.nn.BatchNorm2d(256),
			torch.nn.ReLU(),
			torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
			torch.nn.BatchNorm2d(256),
			torch.nn.ReLU(),
			torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
			torch.nn.BatchNorm2d(256),
			torch.nn.ReLU(),
			torch.nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1),
			torch.nn.BatchNorm2d(128),
			torch.nn.ReLU(),  
			torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
			torch.nn.BatchNorm2d(128),
			torch.nn.ReLU(),
			torch.nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1),
			torch.nn.BatchNorm2d(64),
			torch.nn.ReLU(),
			torch.nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1),
			torch.nn.BatchNorm2d(32),
			torch.nn.ReLU(),
			torch.nn.Conv2d(in_channels=32, out_channels=3, kernel_size=3, stride=1, padding=1),
			torch.nn.Sigmoid()
		)
		
	def forward(self, x):
			
		return self.generator(x)

class Discriminator(torch.nn.Module):
	
	def __init__(self, global_height, global_width, local_height, local_width):
		
		super(Discriminator, self).__init__()
		
		self.global_discriminator = torch.nn.Sequential(

			torch.nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5, stride=2, padding=2),
			torch.nn.BatchNorm2d(64),
			torch.nn.ReLU(),
			torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=2, padding=2),
			torch.nn.BatchNorm2d(128),
			torch.nn.ReLU(),
			torch.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=5, stride=2, padding=2),
			torch.nn.BatchNorm2d(256),
			torch.nn.ReLU(),
			torch.nn.Conv2d(in_channels=256, out_channels=512, kernel_size=5, stride=2, padding=2),
			torch.nn.BatchNorm2d(512),
			torch.nn.ReLU(),
			torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=5, stride=2, padding=2),
			torch.nn.BatchNorm2d(512),
			torch.nn.ReLU(),
			torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=5, stride=2, padding=2)
		)	
		
		self.local_discriminator = torch.nn.Sequential(

			torch.nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5, stride=2, padding=2),
			torch.nn.BatchNorm2d(64),
			torch.nn.ReLU(),		
			torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=2, padding=2),
			torch.nn.BatchNorm2d(128),
			torch.nn.ReLU(),
			torch.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=5, stride=2, padding=2),
			torch.nn.BatchNorm2d(256),
			torch.nn.ReLU(),
			torch.nn.Conv2d(in_channels=256, out_channels=512, kernel_size=5, stride=2, padding=2),
			torch.nn.BatchNorm2d(512),
			torch.nn.ReLU(),
			torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=5, stride=2, padding=2)
		)				
		
		self.local_flatten_num = 512 * int(local_height / 32) * int(local_width / 32) 
		self.global_flatten_num = 512 * int(global_height / 64) * int(global_width / 64) 
		
		self.global_fc = torch.nn.Linear(self.global_flatten_num, 1024)
		self.local_fc = torch.nn.Linear(self.local_flatten_num, 1024)
		self.out_fc = torch.nn.Linear(2 * 1024, 1)
		
		self.train()
	
	def forward(self, x_global, x_local):
		
		x_global = self.global_discriminator(x_global)
		x_local = self.local_discriminator(x_local)
		x_global = x_global.view(-1, self.global_flatten_num)		
		x_local = x_local.view(-1, self.local_flatten_num)
			
		x_global = self.global_fc(x_global)
		x_local = self.local_fc(x_local)	
		x = torch.cat((x_global, x_local), -1)
		
		x = self.out_fc(x)
		y = torch.sigmoid(x)

		return y

class RoiCrop(object):
	
	def __init__(self, batch_size, roicrop_height, roicrop_width, num_gpus):
		
		self.roi_crop = CropAndResize(roicrop_height, roicrop_width)
		boxes_idxs = torch.IntTensor(range(int(batch_size/num_gpus)))
	
		self.gpus_boxes_idxs = []
		for i in range(num_gpus):
			gpu_boxes_idxs = boxes_idxs.cuda(i)
			self.gpus_boxes_idxs.append(gpu_boxes_idxs)
		
	def forward(self, x, boxes, gpu_id):
				
		boxes_idxs = self.gpus_boxes_idxs[gpu_id]
		crops = self.roi_crop(x, boxes, boxes_idxs)
		
		return crops

class Model(torch.nn.Module):
	
	def __init__(self, batch_size, input_height, input_width, roicrop_height, roicrop_width, mean, num_gpus):
		
		super(Model, self).__init__()
		
		self.gpus_means = []
		for i in range(num_gpus):
			gpu_mean = mean.cuda(i)
			self.gpus_means.append(gpu_mean)

		self.generator = Generator()
		self.roicrop = RoiCrop(batch_size, roicrop_height, roicrop_width, num_gpus)
		self.discriminator = Discriminator(input_height, input_width, roicrop_height, roicrop_width)
		
	def forward(self, imgs, masks, completed_boxes=None, image_boxes=None, train=True):
		
		gpu_id = imgs.get_device()		
		imgs = imgs/255.0
		generator_input = torch.cat(((imgs - self.gpus_means[gpu_id]) * (1.0-masks), masks), 1)
		raw_completed = self.generator(generator_input)
		completed_global = raw_completed * masks + imgs * (1.0-masks)				
		
		if train == False:
			
			return completed_global		
		
		else:

			completed_local = self.roicrop.forward(completed_global, completed_boxes, gpu_id)
			imgs_local = self.roicrop.forward(imgs, image_boxes, gpu_id)

			fake = self.discriminator(completed_global, completed_local)
			real = self.discriminator(imgs, imgs_local)

			mse_loss = ((completed_global - imgs)**2).mean()
			gen_loss = torch.log(1.0 - fake).sum()
			dis_loss = (-torch.log(1.0 - fake) - torch.log(real)).sum()
			
			return mse_loss, gen_loss, dis_loss	
