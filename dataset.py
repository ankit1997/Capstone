import os
import glob
import config
import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf
from sklearn.model_selection import train_test_split

def _float_feature(value):
	return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int_feature(value):
	return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
	return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _save_record_file(fname, images, df):
	writer = tf.python_io.TFRecordWriter(fname)
	
	for img_fname in images:
		img = np.array(Image.open(img_fname))
		img_data = df.loc[df.fname == img_fname, ["steer", "indicator"]]

		steer = float(img_data.steer.values[0])
		indicator = int(img_data.indicator.values[0])

		feature = {
			'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.compat.as_bytes(img.tostring())])),
			'steer': tf.train.Feature(float_list=tf.train.FloatList(value=[steer])),
			'indicator': tf.train.Feature(int64_list=tf.train.Int64List(value=[indicator])),
		}

		example = tf.train.Example(features=tf.train.Features(feature=feature))
		writer.write(example.SerializeToString())

	writer.close()
	print("Saved to {} successfully.".format(fname))

def save_tfrecords():

	# Read the csv file with steering and indicator values
	driving_log = pd.read_csv(config.driving_log_fname)

	images = driving_log.fname.values
	assert all(os.path.isfile(img_file) for img_file in images) == True

	X_train, X_test = train_test_split(images, test_size=0.1)
	X_train, X_valid = train_test_split(X_train, test_size=0.1)

	num_train_batches = np.ceil(X_train.shape[0] / config.batch_size)
	num_valid_batches = np.ceil(X_valid.shape[0] / config.batch_size)
	num_test_batches = np.ceil(X_test.shape[0] / config.batch_size)

	with open(config.meta_dataset, 'w') as f:
		f.write("{},{},{}".format(num_train_batches, num_valid_batches, num_test_batches))

	_save_record_file(config.train_tfrecord_fname, X_train, driving_log)
	_save_record_file(config.valid_tfrecord_fname, X_valid, driving_log)
	_save_record_file(config.test_tfrecord_fname, X_test, driving_log)

def augment(imgs, steers, indicators):

	# horizontal flips
	num = tf.random_uniform((1,))[0]
	imgs, steers, indicators = tf.cond(num>0.5, 
								true_fn=lambda: [tf.map_fn(tf.image.flip_left_right, imgs), -steers, -indicators], 
								false_fn=lambda: [imgs, steers, indicators])

	# brightness
	num = tf.random_uniform((1,))[0]
	imgs, steers, indicators = tf.cond(num>0.5, 
								true_fn=lambda: [tf.image.adjust_brightness(imgs, 0.5), steers, indicators], 
								false_fn=lambda: [imgs, steers, indicators])

	# contrast
	num = tf.random_uniform((1,))[0]
	imgs, steers, indicators = tf.cond(num>0.5, 
								true_fn=lambda: [tf.image.adjust_contrast(imgs, 0.5), steers, indicators], 
								false_fn=lambda: [imgs, steers, indicators])

	# translate
	num = tf.random_uniform((1,))[0]
	translations = tf.random_uniform(shape=(config.batch_size, 2), minval=?, maxval=?, dtype=tf.int32)
	imgs, steers, indicators = tf.cond(num>0.5, 
								true_fn=lambda: [tf.contrib.image.translate(imgs, translations), steers, indicators], 
								false_fn=lambda: [imgs, steers, indicators])

	
	return [imgs, steers, indicators]

def read_tfrecord(fname):

	filename_queue = tf.train.string_input_producer([fname])

	reader = tf.TFRecordReader()
	_, serialized_example = reader.read(filename_queue)
	
	feature_set = {
		'image': tf.FixedLenFeature([], tf.string),
	    'steer': tf.FixedLenFeature([], tf.float32),
	    'indicator': tf.FixedLenFeature([], tf.int64)
	}

	# Decode the record read by the reader
	features = tf.parse_single_example(serialized_example, features=feature_set)
	
	# Convert the image data from string, back to numbers
	img = tf.decode_raw(features['image'], tf.uint8)
	img = tf.reshape(img, config.IMG_SIZE)

	# Cast steer and indicator
	steer = tf.cast(features['steer'], tf.float32)
	indicator = tf.cast(features['indicator'], tf.int32)

	# Create batches
	imgs, steers, indicators = tf.train.shuffle_batch([img, steer, indicator], 
												batch_size=config.batch_size, 
												capacity=30, 
												min_after_dequeue=10)

	# Augmentation
	true_fn = lambda: augment(imgs, steers, indicators)
	false_fn = lambda: [imgs, steers, indicators]
	num = tf.random_uniform((1,))[0]
	imgs, steers, indicators = tf.cond(num>0.5, true_fn=true_fn, false_fn=false_fn)

	return imgs, steers, indicators

# Testing code
if __name__ == "__main__":
	save_tfrecords()
	imgs, steers, indicators = read_tfrecord(config.train_tfrecord_fname)

	with tf.Session() as sess:
	    coord = tf.train.Coordinator()
	    threads = tf.train.start_queue_runners(coord=coord)
	    for i in range(10):
	        print(i, sess.run([steers]))
	    coord.request_stop()
	    coord.join(threads)