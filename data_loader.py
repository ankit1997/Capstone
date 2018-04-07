import os
import numpy as np
import pandas as pd
from PIL import Image

IMGS_DIRNAME = "IMGS"
STEER_FNAME = "driving_log.csv"

np.random.seed(7)

class DataLoader:
	def __init__(self, path, split=(0.8, 0.1, 0.1), batch_size=16):
		self.batch_size = batch_size
		
		steer_file = os.path.join(path, STEER_FNAME)
		df = pd.read_csv(steer_file).sample(frac=1).reset_index(drop=True)

		self.imgs_dir = os.path.join(path, IMGS_DIRNAME)

		self.num = df.shape[0]
		train_ind = int(self.num * split[0])
		valid_ind = int(self.num * (split[0]+split[1]))
		test_ind = self.num

		self.train = df[:train_ind]
		self.valid = df[train_ind: valid_ind]
		self.test = df[valid_ind:]

		self.num_train_batches = np.ceil(self.train.shape[0]/batch_size)
		self.num_validation_batches = np.ceil(self.valid.shape[0]/batch_size)
		self.num_test_batches = np.ceil(self.test.shape[0]/batch_size)

	def train_generator(self):
		for i in range(self.num_train_batches):
			start = i
			end = min(start + self.batch_size, self.train.shape[0])

			img_fnames = self.train[start: end][0]
			steer_values = np.asarray(self.train[start: end][1]).reshape((-1, 1))

			imgs = [np.expand_dims(np.asarray(Image.open(img_file)), 0) for img_file in img_fnames]
			imgs = np.concatenate(imgs)

			yield imgs, steer_values

	def validation_generator(self):
		for i in range(self.num_validation_batches):
			start = i
			end = min(start + self.batch_size, self.valid.shape[0])
			
			img_fnames = self.valid[start: end][0]
			steer_values = np.asarray(self.valid[start: end][1]).reshape((-1, 1))

			imgs = [np.expand_dims(np.asarray(Image.open(img_file)), 0) for img_file in img_fnames]
			imgs = np.concatenate(imgs)

			yield imgs, steer_values

	def test_generator(self):
		for i in range(self.num_test_batches):
			start = i
			end = min(start + self.batch_size, self.test.shape[0])
			
			img_fnames = self.test[start: end][0]
			steer_values = np.asarray(self.test[start: end][1]).reshape((-1, 1))

			imgs = [np.expand_dims(np.asarray(Image.open(img_file)), 0) for img_file in img_fnames]
			imgs = np.concatenate(imgs)

			yield imgs, steer_values