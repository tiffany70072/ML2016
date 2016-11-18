from numpy import genfromtxt, array
from math  import log
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D, UpSampling2D, Input
from keras.callbacks import EarlyStopping
from keras.regularizers import l2
from keras.backend.tensorflow_backend import set_session
from keras.optimizers import adam
from scipy.spatial import distance
from keras import backend as K
import tensorflow as tf
import numpy as np
import pickle
import sys
import os
import theano

'''config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5
set_session(tf.Session(config = config))'''

os.environ["THEANO_FLAGS"] = "device=gpu0"

def shuffle_in_unison(a, b):
    assert len(a) == len(b)
    shuffled_a = np.empty(a.shape, dtype = a.dtype)
    shuffled_b = np.empty(b.shape, dtype = b.dtype)
    permutation = np.random.permutation(len(a))
    for old_index, new_index in enumerate(permutation):
        shuffled_a[new_index] = a[old_index]
        shuffled_b[new_index] = b[old_index]
    return shuffled_a, shuffled_b

def load_label_data():
	print "Loading Label Data"
	train = pickle.load(open(sys.argv[1]+"/all_label.p", "rb"))
	print "Finished Loading Label Data"

	x_raw = train[0]
	for i in np.arange(1, 10): x_raw = np.concatenate((x_raw, train[i]), axis = 0)
	x_raw = array(x_raw)

	temp = np.empty([3, 32, 32])
	x_train = np.empty([5000, 3, 32, 32])
	for i in range(5000):
		for j in range(3):
			for k in range(32):
				for l in range(32):
				    temp[j][k][l] = x_raw[i][1024*j+32*k+l]
		x_train[i] = temp

	y_train = np.empty([10*500, 10], dtype = int)
	for i in range(10):
		for j in range(500):
			for k in range(10):
				y_train[i*500+j][k] = 0
			y_train[i*500+j][i] = 1

	#pickle.dump(x_train, open("model-x-train", "wb")) 
	#pickle.dump(y_train, open("model-y-train", "wb"))
	#x_train = pickle.load(open("model-x-train", "rb"))
	#y_train = pickle.load(open("model-y-train", "rb")) 

	print "x.shape = ", x_train.shape
	print "y.shape = ", y_train.shape
	print "Finished Data Preprocessing"
	return x_train, y_train

def load_unlabel_data():
	print "Loading Unlabel Data"
	unlabel = pickle.load(open(sys.argv[1]+"/all_unlabel.p", "rb"))
	print "Finished Loading Unlabeled Data"

 	unlabel = array(unlabel)
	temp = np.empty([3, 32, 32])
	x_u = np.empty([45000, 3, 32, 32])
	for i in range(45000):
		for j in range(3):
			for k in range(32):
				for l in range(32):
				    temp[j][k][l] = unlabel[i][1024*j+32*k+l]
		x_u[i] = temp
	#pickle.dump(x_u, open("model-x-unlabel", "wb")) 
	#x_u = pickle.load(open("model-x-unlabel", "rb"))

	print "x_u.shape = ", x_u.shape
	return x_u
	
def load_test_data():
	print "Finished Training, Start Loading Testing Data"
	test = pickle.load(open(sys.argv[1]+"/test.p", "rb"))
	print "Finished Loading"

	test_raw = array(test.get("data"))
	temp = np.empty([3, 32, 32])
	x_test = np.empty([10000, 3, 32, 32])
	for i in range(10000):
		for j in range(3):
			for k in range(32):
				for l in range(32):
					temp[j][k][l] = test_raw[i][1024*j+32*k+l]
		x_test[i] = temp
	x_test /= 255.0
	#pickle.dump(x_test, open("model-x-test", "wb")) 
	#x_test = pickle.load(open("model-x-test", "rb"))

	print "x_test.shape = ", x_test.shape
	return x_test

def output_result(result): # write file
	print result.shape
	fout = open("result-eating.csv", 'w')
	fout.write("ID,class\n")

	for i in range(10000):
		maxP = 0.0
		maxId = -1
		for j in range(10):
			if result[i][j] > maxP:
				maxP = result[i][j]
				maxId = j
		fout.write("%d,%d\n" % (i, maxId))

def construct_model(model):
	model.add(Convolution2D(25, 3, 3, dim_ordering = "th", input_shape = (3, 32, 32)))
	model.add(MaxPooling2D((2, 2), dim_ordering = "th"))
	model.add(Convolution2D(50, 3, 3, dim_ordering = "th"))
	model.add(MaxPooling2D((2, 2), dim_ordering = "th"))
	model.add(Flatten())
	model.add(Dense(output_dim = 700, W_regularizer = l2(0.01)))
	model.add(Activation('sigmoid'))
	#model.add(Dropout(0.25))
	#model.add(Dense(output_dim = 250))#, W_regularizer = l2(0.03)))
	#model.add(Activation('sigmoid'))
	#model.add(Dropout(0.25))
	model.add(Dense(output_dim = 10))
	model.add(Activation('softmax'))

def divide(x_label, y_label, n):
	x_label, y_label = shuffle_in_unison(x_label, y_label)
	x_label, y_label = shuffle_in_unison(x_label, y_label)
	print "Dividing Data"
	x_valid = x_label[:n, :]
	x_train = x_label[n:, :]
	y_valid = y_label[:n, :]
	y_train = y_label[n:, :]
	print "divide = ", x_train.shape, x_valid.shape, y_valid.shape
	return x_train, x_valid, y_train, y_valid

def cv_error(model, x_valid, y_valid):
	score = model.evaluate(x_valid, y_valid)
	print "\nTotal loss on CV set: ", score[0]
	print "Accuracy on CV set: ", score[1]

def entropy(y):
	ent = 0
	for i in range(10): 
		if y[i] != 0.0: ent -= y[i]*log(y[i])
	return ent

def similarity(unlabel, label):
	dst = distance.euclidean(unlabel, label)
	return dst

def label_data(x_u, x_label, encoder, threshold):
	newId = []
	newLabel = []
	count = 0
	for i in range(45000):
		unlabel = encoder.predict(x_u[i:i+1])
		unlabel = unlabel.flatten()
		minId = -1
		minDis = threshold
		if (i<10000 and i>9990) or (i<30000 and i>29990): print "\n", i, " - ",
		for j in range(10):
			distance = similarity(unlabel, x_label[j])
			if (i<10000 and i>9990) or (i<30000 and i>29990): print "%.3f, " %distance, 
			if distance < minDis:
				minDis = distance
				minId = j
		if minId != -1: 
			newId.append(i)
			newLabel.append(minId)
			count += 1

	x_new = np.empty([count, 3, 32, 32], dtype = float)
	y_new = np.zeros([count, 10], dtype = int)		
	for i in range(count):
		x_new[i] = x_u[newId[i]]
		y_new[i][newLabel[i]] = 1

	print "size of new data = ", count
	print "x_new = ", x_new.shape
	print "y_new = ", y_new.shape
	print "Finished Labeling Unlabeled Data." 
	return x_new, y_new

def merge(x_t, y_t, x_n, y_n):
	x = np.concatenate((x_t, x_n), axis = 0)
	y = np.concatenate((y_t, y_n), axis = 0)
	print "label = ", y.shape[0]
	return x, y

def test(model):
	x_test = load_test_data()
	result = model.predict(x_test)
	output_result(result)

def construct_autoencoder():
	inputs = Input(shape = (3, 32, 32))
	x = Convolution2D(16, 3, 3,    border_mode = 'same', dim_ordering = "th", activation = 'relu')(inputs)
	x = MaxPooling2D((2, 2),	   border_mode = 'same', dim_ordering = "th")(x)
	x = Convolution2D(16, 3, 3,    border_mode = 'same', dim_ordering = "th", activation = 'relu')(x)
	encoded = MaxPooling2D((4, 4), border_mode = 'same', dim_ordering = "th")(x)

	x = Convolution2D(16, 3, 3, border_mode = 'same', dim_ordering = "th", activation = 'relu')(encoded)
	x = UpSampling2D((4, 4),    dim_ordering = "th")(x)
	x = Convolution2D(16, 3, 3, border_mode = 'same', dim_ordering = "th", activation = 'relu')(x)
	x = UpSampling2D((2, 2),    dim_ordering = "th")(x)
	decoded = Convolution2D(3, 3, 3, border_mode = 'same', dim_ordering = "th", activation = 'sigmoid')(x)

	autoencoder = Model(input = inputs, output = decoded)
	autoencoder.summary()
	#autoencoder.compile(optimizer = 'adadelta', loss = 'binary_crossentropy')
	autoencoder.compile(optimizer = 'adam', loss = 'binary_crossentropy')
	x_noisy = 0.9 * x_label + 0.1 * np.random.normal(loc = 0.0, scale = 1.0, size = x_label.shape) 
	autoencoder.fit(x_noisy, x_label, nb_epoch = 10, batch_size = 500, shuffle = True)

	return inputs, encoded, decoded

def encode_label_data(x_label):
	print "Encoder"
	N = 256
	x_encoded = np.empty([10, N])
	temp2 = np.empty([500, N])
	for i in range(10):
		#total = 0
		temp = encoder.predict(x_label[i*500:(i+1)*500])
		for j in range(500): temp2[j] = temp[j].flatten()
		x_encoded[i] = sum(temp2)/500

	print x_encoded.shape
	for i in range(10):
		for j in range(10):
			print "%.3f, " %x_encoded[i][j],
		print ""
	return x_encoded

def train(x_label, y_label, x_new, y_new):
	print "After Labeling, and Start training"
	#x_label, x_valid, y_label, y_valid = divide(x_label, y_label, 1000)
	x_label, y_label = merge(x_label, y_label, x_new, y_new)
	model = Sequential()
	construct_model(model)
	model.summary()
	model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
	model.fit(x_label, y_label, batch_size = y_label.shape[0]/5, nb_epoch = 200, validation_data = None, shuffle = True)
	#cv_error(model, x_valid, y_valid)200

	return model

x_label, y_label = load_label_data()
x_label /= 255.0
x_u = load_unlabel_data()
x_u /= 255.0

inputs, encoded, decoded = construct_autoencoder()
encoder = Model(input = inputs, output = encoded)
x_encoded = encode_label_data(x_label)
x_new, y_new = label_data(x_u, x_encoded, encoder, 1)

model = train(x_label, y_label, x_new, y_new)
#test(model)
model.save(sys.argv[2])
#pickle.dump(model, open(sys.argv[2], "wb")) 

'''model = Sequential()
model.add(Convolution2D(32, 3, 3, activation = 'relu', border_mode = 'same', dim_ordering = "th", input_shape = (3, 32, 32)))
model.add(MaxPooling2D((2, 2), border_mode = 'same', dim_ordering = "th"))
model.add(Convolution2D(16, 3, 3, activation = 'relu', border_mode = 'same', dim_ordering = "th"))
encoded = MaxPooling2D((4, 4), border_mode = 'same', dim_ordering = "th")
model.add(encoded)
model.add(Convolution2D(16, 3, 3, activation = 'relu', border_mode = 'same', dim_ordering = "th"))
model.add(UpSampling2D((4, 4), dim_ordering = "th"))
model.add(Convolution2D(32, 3, 3, activation = 'relu', border_mode = 'same', dim_ordering = "th"))
model.add(UpSampling2D((2, 2), dim_ordering = "th"))
model.add(Convolution2D(3, 3, 3, activation = 'sigmoid', border_mode = 'same', dim_ordering = "th"))
model.summary()

model.compile(optimizer = 'adadelta', loss = 'binary_crossentropy')
x_noisy = 0.7 * x_label + 0.3 * np.random.normal(loc = 0.0, scale = 1.0, size = x_label.shape) 
model.fit(x_noisy, x_label, nb_epoch=1, batch_size=1000, shuffle=True)
out = K.function([model.layers[0].input], [model.layers[3].output])

print out([x_label[0]])[0]'''

