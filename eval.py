import torch
import numpy as np
import cv2
import network

GENERATOR_WEIGHTS = './model/generator-1000.pt'
IMG_FILEPATH = './street.jpg'
RESULT_FILEPATH = './result.png'
RGB_MEAN = np.array([0.4560, 0.4472, 0.4155])

generator = network.Generator()
generator.load_state_dict(torch.load(GENERATOR_WEIGHTS))
generator.cuda()
generator.eval()	

img = cv2.imread(IMG_FILEPATH)
height, width, _ = img.shape
height = height - height%4
width = width - width%4
img = img[:height, :width, :]

img = np.array(img[...,::-1])
mask = np.zeros((1, height, width))	
y1 = int(0.25 * height)
y2 = int(0.75 * height)
x1 = int(0.25 * width)
x2 = int(0.75 * width)
mask[:, y1: y2, x1: x2] = 1.0

with torch.no_grad():
	
	img =  torch.FloatTensor(np.expand_dims(img, 0)).cuda()
	mask = torch.FloatTensor(np.expand_dims(mask, 0)).cuda()
	mean = torch.FloatTensor(RGB_MEAN).cuda()
	
	img = img/255.0
	img_input = (img - mean).permute(0, 3, 1, 2)
	img_oryginal = img.permute(0, 3, 1, 2)
	generator_input = torch.cat((img_input *(1.0 - mask), mask), 1)
	raw_completed = generator(generator_input)
	completed_global = raw_completed*mask + img_oryginal * (1.0-mask)
	completed_global = completed_global * 255.0
	
	img = completed_global.data.cpu().numpy()[0]
	img = img.transpose(1, 2, 0)
	img = img[...,::-1].astype(np.uint8)

	cv2.imwrite(RESULT_FILEPATH, img)
	
	print 'evaluation done'
