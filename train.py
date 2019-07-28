import torch
import numpy as np
import cv2, os, shutil, sys
import network
from torch.utils.data import Dataset

BATCH_SIZE = 12
NUM_GPUS = torch.cuda.device_count()
IMG_HEIGHT, IMG_WIDTH = 256, 256
ROICROP_HEIGHT, ROICROP_WIDTH = 128, 128
EPOCH_NUM = 100000
RGB_MEAN = [0.4560, 0.4472, 0.4155]
TRAIN_IMGS_DIR = './train_imgs'
TEST_IMGS_DIR = './test_imgs'
TEST_RESULT_DIR = './test_result'
MODEL_SAVE_DIR = './model'

T_D = 10
T_C = 10
ALPHA = 0.0004
LOAD_PRETRAINED_GENERATOR = True
LOAD_PRETRAINED_DISCRIMINATOR = True
PRETRAINED_DISCRIMINATOR_WEIGHTS = './pretrained/discriminator.pt'
PRETRAINED_GENERATOR_WEIGHTS = './pretrained/generator.pt'

class ImgDataset(Dataset):
	
	def __init__(self, imgs_dir, height, width, train=True):
		
		self.train = train
		
		self.images_names = sorted(os.listdir(imgs_dir))
		self.images_paths = map(lambda image_name: os.path.join(imgs_dir, image_name), self.images_names)
		
		if len(self.images_paths) == 0:
			print 'Images directory is empty.'
			sys.exit()

		self.width = width
		self.height = height
		
	def __len__(self):
		
		return len(self.images_paths)
	
	def random_box(self):
		
		if self.train == False:
			
			w = 0.5
			h = 0.5
			x1 = 0.25
			y1 = 0.25		

		else:
			
			w = np.random.uniform(0.3, 0.7)
			h = np.random.uniform(0.3, 0.7)
			x1 = np.random.uniform(0.1, 0.9 - w)
			y1 = np.random.uniform(0.1, 0.9 - h)			

		x2 = x1 + w
		y2 = y1 + h

		return y1, x1, y2, x2
						
	def __getitem__(self, idx):
		
		image = cv2.imread(self.images_paths[idx])
		image = cv2.resize(image, (self.width, self.height))
		image = image[...,::-1]
		
		# generate mask as 4-th channel for generator and discriminator input
		y1, x1, y2, x2 = self.random_box()
		mask = np.zeros((1, image.shape[0], image.shape[1]))	
		mask[:, int(y1*self.height):int(y2*self.height), int(x1*self.width):int(x2*self.width)] = 1.0

		# add margins
		y1 = y1 - 0.05
		x1 = x1 - 0.05
		y2 = y2 + 0.05
		x2 = x2 + 0.05
		inpainted_box = [y1, x1, y2, x2]

		y1, x1, y2, x2 = self.random_box()
		random_box = [y1, x1, y2, x2]

		# convert to pytorch tensors
		image = image.transpose(2, 0, 1).astype(np.float32)
		image =  torch.FloatTensor(image)
		mask = torch.FloatTensor(mask)
		inpainted_box = torch.FloatTensor(inpainted_box)
		random_box = torch.FloatTensor(random_box)

		return image, mask, inpainted_box, random_box

train_dataset = ImgDataset(TRAIN_IMGS_DIR, IMG_HEIGHT, IMG_WIDTH)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

test_dataset = ImgDataset(TEST_IMGS_DIR, IMG_HEIGHT, IMG_WIDTH, train=False)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

mean = torch.FloatTensor(RGB_MEAN)
mean = mean.repeat(1, IMG_HEIGHT, IMG_WIDTH, 1)
mean = mean.permute(0, 3, 1, 2)

model = network.Model(BATCH_SIZE, IMG_HEIGHT, IMG_WIDTH, ROICROP_HEIGHT, ROICROP_WIDTH, mean, NUM_GPUS)

if LOAD_PRETRAINED_GENERATOR == True:
	
	T_C = 0 # do not pretrain generator with mse loss
	model.generator.load_state_dict(torch.load(PRETRAINED_GENERATOR_WEIGHTS))
	
if LOAD_PRETRAINED_DISCRIMINATOR == True:
	
	T_D = 0 # do not pretrain discriminator
	model.discriminator.load_state_dict(torch.load(PRETRAINED_DISCRIMINATOR_WEIGHTS))

model.cuda()
model = torch.nn.DataParallel(model)

gen_optimizer = torch.optim.Adadelta(model.module.generator.parameters())
dis_optimizer = torch.optim.Adadelta(model.module.discriminator.parameters())

def make_dir(directory):
	if os.path.exists(directory):
		shutil.rmtree(directory, ignore_errors=True)
	os.makedirs(directory)

make_dir(TEST_RESULT_DIR)
make_dir(MODEL_SAVE_DIR)

global_iteration = 0
model.module.generator.train()

for epoch_iteration in range(EPOCH_NUM):

	print 'epoch iteration:', epoch_iteration

	for iteration, batch_data in enumerate(train_dataloader):
		
		global_iteration = global_iteration + 1;
		
		images, masks, inpainted_boxes, random_boxes = batch_data
		mse_loss, gen_loss, dis_loss = model(images, masks, inpainted_boxes, random_boxes)
		
		# ------------------------------------------------------------------------	
		# ------------------- generator training with mse loss -------------------
		# ------------------------------------------------------------------------
					
		if global_iteration < T_C: 
			
			mse_loss = mse_loss.sum()
			gen_optimizer.zero_grad()
			mse_loss.backward()
			gen_optimizer.step()				
			
			mse_loss_data = mse_loss.data.cpu().numpy()	
			
			print 'iter:', global_iteration, 'mse loss:', mse_loss_data
		
		else:
			
			# ------------------------------------------------------------------------	
			# ------------------------ discriminator training ------------------------
			# ------------------------------------------------------------------------
					
			if global_iteration < T_D + T_C: 

				dis_loss = dis_loss.sum()
				dis_loss = ALPHA * dis_loss
				dis_optimizer.zero_grad()
				dis_loss.backward()
				dis_optimizer.step()
				
				dis_loss_data = dis_loss.data.cpu().numpy()				
				
				print 'iter:', global_iteration, 'dis loss:', dis_loss_data
							
			# ------------------------------------------------------------------------	
			# ----------------- generator and discriminator training -----------------
			# ------------------------------------------------------------------------
					
			else:
				
				mse_loss = mse_loss.sum()
				gen_loss = gen_loss.sum()
				dis_loss = dis_loss.sum()
				gen_loss = mse_loss + ALPHA * gen_loss
				dis_loss = ALPHA * dis_loss
				dis_optimizer.zero_grad()
				dis_loss.backward(retain_graph=True)
				dis_optimizer.step()
				gen_optimizer.zero_grad()
				gen_loss.backward()
				gen_optimizer.step()	
				
				mse_loss_data = mse_loss.data.cpu().numpy()				
				gen_loss_data = gen_loss.data.cpu().numpy()
				dis_loss_data = dis_loss.data.cpu().numpy()
				
				print 'iter:', global_iteration, 'mse_loss:', mse_loss_data, 'gen_loss:', gen_loss_data, 'dis loss:', dis_loss_data
			
		# ------------------------------------------------------------------------	
		# ------------------------------ save model ------------------------------
		# ------------------------------------------------------------------------
					
		if global_iteration % 10 == 0:
			
			generator_filepath = MODEL_SAVE_DIR + '/' + 'generator-' + str(global_iteration) + '.pt'
			discriminator_filepath = MODEL_SAVE_DIR + '/' + 'discriminator-' + str(global_iteration) + '.pt'

			generator_cpu = model.module.generator.cpu()
			discriminator_cpu = model.module.discriminator.cpu()
						
			torch.save(generator_cpu.state_dict(), generator_filepath)
			torch.save(discriminator_cpu.state_dict(), discriminator_filepath)
	
			model.cuda()
			
			print 'iter:', global_iteration, 'model saved'
			
		# ------------------------------------------------------------------------	
		# -------------------------- evaluate generator --------------------------
		# ------------------------------------------------------------------------
							
		if global_iteration % 10 == 0 and global_iteration >= T_D + T_C:
			
			model.module.generator.eval()	
			
			for eval_iteration, batch_data in enumerate(test_dataloader):
				
				with torch.no_grad():
					
					images, masks, _, _ = batch_data

					completed_global = model(images, masks, train=False)
					data_completed_global = completed_global.data.cpu().numpy()
					data_completed_global = data_completed_global * 255.0
				
					for i, img in enumerate(data_completed_global):
						img = img.transpose(1, 2, 0)
						img = img[...,::-1].astype(np.uint8)
						
						file_name = 'iter:' + str(global_iteration) + '-img:' + str(eval_iteration*BATCH_SIZE + i) + '.png'
						
						cv2.imwrite(TEST_RESULT_DIR + '/' + file_name, img)

			model.module.generator.train()
					
			print 'iter:', global_iteration, 'evaluation done'
