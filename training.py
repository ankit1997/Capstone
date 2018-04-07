import os
from tqdm import tqdm
from data_loader import DataLoader
from model import build_model, IMG_SHAPE

MODEL_DIR = "saved_model"

model_file = os.path.join(MODEL_DIR, "model.ckpt")
LVE_file = os.path.join(MODEL_DIR, "lve.ckpt") # least validation error param file.
os.makedirs(MODEL_DIR, exist_ok=True)

LVE_val = os.path.join(MODEL_DIR, "lve_val.txt") # to store lve

def train(epochs, logging=10, checkpoint=20, resume=False):
	
	# Get the model
	inp_img, out_steer, pred_steer, loss, train_step = build_model()

	# Get the data loader
	dataLoader = DataLoader(split=(0.8, 0.1, 0.1))

	print("Beginning training...")

	session = tf.Session()
	saver = tf.train.Saver()
	MIN_VALID_LOSS = 100000000.0
	if os.path.isfile(LVE_val):
		try:
			MIN_VALID_LOSS = float(open(LVE_val).read().strip())
		except:
			print("Invalid data in {}.".format(LVE_val))
	else:
		with open(LVE_val, 'w') as f:
			f.write(str(MIN_VALID_LOSS))

	if resume:
		try:
			saver.restore(session, model_file)
			print("Model restore successful :)")
		except:
			print("Model restore unsuccessful :(")

	for e in tqdm(range(1, epochs+1)):
		sumL = 0.0

		for batch_x, batch_y in dataLoader.train_generator():
			L, _ = session.run([loss, train_step], 
								feed_dict={inp_img: batch_x, out_steer: batch_y})
			sumL += L
		avgL = sumL/dataLoader.num_train_batches
		print("Epoch {} : Average training loss = {}".format(e, avgL))

		if e%logging == 0:
			
			# Run validation set
			vsumL = 0.0

			for vbatch_x, vbatch_y in dataLoader.validation_generator():
				L = session.run(loss, feed_dict={inp_img: batch_x, out_steer: batch_y})
				vsumL += L
			vavgL = vsumL/dataLoader.num_validation_batches

			print("Average validation loss = {}".format(vavgL))

			if vavgL < MIN_VALID_LOSS:
				saver.save(session, LVE_file)
				print("Saved lve parameters at epoch {}".format(e))

				MIN_VALID_LOSS = vavgL
				with open(LVE_val, 'w') as f:
					f.write(str(MIN_VALID_LOSS))

			print("---------------------------------------------------------------")

		if e%checkpoint == 0:
			saver.save(session, model_file)
			print("Saved model at epoch {}".format(e))

	# Run test set
	sumL = 0.0

	for batch_x, batch_y in dataLoader.test_generator():
		L = session.run(loss, feed_dict={inp_img: batch_x, out_steer: batch_y})
		sumL += L
	avgL = sumL/dataLoader.num_test_batches

	print("Average testing loss = {}".format(avgL))

	session.close()