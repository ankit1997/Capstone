import tensorflow as tf
print("Using tensorflow: {}".format(tf.__version__))

IMG_SHAPE = (66, 160, 3)
ELU = tf.nn.elu

def build_model():
	print("Building model...")

	# Define placeholders
	img = tf.placeholder(tf.uint8, shape=(None, *IMG_SHAPE), name="Input-image") # values between 0-255
	indicator = tf.placeholder(tf.float32, shape=(None, 1), name='Indicator-flag') # values {-1.0, 0.0, 1.0}
	steer = tf.placeholder(tf.float32, shape=(None, 1), name="Output-steer") # values [-1.0, 1.0]

	# Image normalization [-1.0, 1.0]
	img = tf.cast(img, tf.float32)	
	img = img / 127.5 - 1.0

	# Convolution layers

	print("Img: {}".format(img.shape))

	conv1 = tf.layers.conv2d(img, 
							filters=32, 
							kernel_size=5, 
							strides=2, 
							padding="valid", 
							activation=ELU,
							name='Convolution-1')

	pool1 = tf.layers.max_pooling2d(conv1, 
									pool_size=2, 
									strides=2,
									name='Pooling-1')

	print("Pool1: {}".format(pool1.shape))

	conv2 = tf.layers.conv2d(conv1, 
							filters=64, 
							kernel_size=5, 
							strides=2, 
							padding="valid", 
							activation=ELU,
							name='Convolution-2')

	pool2 = tf.layers.max_pooling2d(conv2, 
									pool_size=2, 
									strides=2,
									name='Pooling-1')

	print("Pool2: {}".format(pool2.shape))

	# Flatten output of conv layers
	convn = pool2
	flatten = tf.layers.flatten(convn, name="Flatten")

	print("Flatten: {}".format(flatten.shape))

	# Dense layers
	dense1 = tf.layers.dense(flatten, units=256, activation=ELU, name="Dense-1")
	drop1 = tf.layers.dropout(dense1, rate=0.3, name="Dropout-1")

	print("Dense1: {}".format(drop1.shape))

	dense2 = tf.layers.dense(drop1, units=64, activation=ELU, name="Dense-2")
	drop2 = tf.layers.dropout(dense2, rate=0.3, name="Dropout-2")

	print("Dense2: {}".format(drop2.shape))

	dense3 = tf.layers.dense(drop2, units=7, activation=ELU, name="Dense-3")
	drop3 = tf.layers.dropout(dense3, rate=0.3, name="Dropout-3")
	drop3_withIndicator = tf.concat([drop3, indicator], axis=1, name="Add-meta-data") # merge indicator placeholder

	print("Dense3: {}".format(drop3_withIndicator.shape))
	
	dense4 = tf.layers.dense(drop3_withIndicator, units=4, activation=ELU, name="Dense-4")
	print("Dense4: {}".format(dense4.shape))

	dense5 = tf.layers.dense(dense4, units=1, activation=tf.nn.tanh, name="Dense-5")
	print("Dense5: {}".format(dense5.shape))

	pred = dense5

	print("Pred: {}".format(pred.shape))
	
	# Define loss
	loss = tf.losses.mean_squared_error(labels=steer, predictions=pred)

	# Define train step
	train_step = tf.train.AdamOptimizer(0.0001).minimize(loss)

	return img, steer, indicator, pred, loss, train_step

if __name__ == "__main__":
	inp_img, inp_steer, inp_indicator, pred_steer, loss, train_step = build_model()