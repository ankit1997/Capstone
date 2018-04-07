import tensorflow as tf

IMG_SHAPE = (100, 100, 3)
ELU = tf.nn.elu
CONV_PARAMS = [
	{'filters': , 'kernel_size': , 'padding': , },
]

def build_model():
	print("Building model...")

	# Define placeholders
	img = tf.placeholder(tf.uint8, shape=(None, *IMG_SHAPE), name="Input-image")
	steer = tf.placeholder(tf.float32, shape=(None, 1), name="Output-steer")

	# Image normalization [-1.0, 1.0]
	img = img / 127.5 - 1.0

	# Convolution layers
	conv1 = tf.layers.conv2d(img, filters=, kernel_size=[], padding="", activation=ELU)
	pool1 = tf.layers.max_pooling2d(conv1, pool_size=2, strides=)
	conv1 = tf.layers.conv2d(img, filters=, kernel_size=[], padding="", activation=ELU)
	pool2 = tf.layers.max_pooling2d(conv2, pool_size=2, strides=)

	# Flatten output of conv layers
	convn = pool2
	flatten = tf.layers.flatten(convn, name="Flatten")

	# Dense layers
	dense1 = tf.layers.dense(flatten, units=, activation=)
	drop1 = tf.layers.dropout(dense1, rate=0.3)

	dense2 = tf.layers.dense(drop1, units=, activation=)
	drop2 = tf.layers.dropout(dense2, rate=0.3)
	
	dense3 = tf.layers.dense(drop2, units=, activation=)

	pred = dense3
	
	# Define loss
	loss = tf.losses.mean_squared_error(labels=steer, predictions=pred)

	# Define train step
	train_step = tf.train.AdamOptimizer(0.0001).minimize(loss)

	return img, steer, pred, loss, train_step

if __name__ == "__main__":
	inp_img, out_steer, pred_steer, loss, train_step = build_model()