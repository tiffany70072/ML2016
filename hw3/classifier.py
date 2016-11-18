from numpy import genfromtxt, array
from math import log
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Convolution2D, MaxPooling2D
from keras.callbacks import EarlyStopping
from keras.regularizers import l2
from keras.optimizers import Adam
from keras.backend.tensorflow_backend import set_session

import tensorflow as tf
import numpy as np
import pickle
import sys
import os

'''config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.7
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
	
	#pickle.dump(x_test, open("model-x-test", "wb")) 
	#x_test = pickle.load(open("model-x-test", "rb"))

	print "x_test.shape = ", x_test.shape
	return x_test

def output_result(result): # write file
	print result.shape
	fout = open("result-eating.csv", 'w')
	'''if i == 0: 
	elif i == 1: fout = open("result-test3.csv", 'w')
	elif i == 2: fout = open("result-test4.csv", 'w')
	elif i == 3: fout = open("result-test5.csv", 'w')
	elif i == 4: fout = open("result-test6.csv", 'w')
	else : fout = open("result-test7.csv", 'w')'''

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
	model.add(MaxPooling2D((2, 2),    dim_ordering = "th"))
	model.add(Convolution2D(50, 3, 3, dim_ordering = "th"))
	model.add(MaxPooling2D((2, 2),    dim_ordering = "th"))
	model.add(Flatten())
	model.add(Dense(output_dim = 500, W_regularizer = l2(0.03)))
	#model.add(Activation('relu'))
	#model.add(Dropout(0.25))
	#model.add(Dense(output_dim = 128, W_regularizer = l2(0.01)))
	model.add(Activation('sigmoid'))
	model.add(Dropout(0.4))
	model.add(Dense(output_dim = 10))
	model.add(Activation('softmax'))

def construct_model2(model, dropout):
	print "Dropout = ", dropout
	model.add(Convolution2D(25, 3, 3, dim_ordering = "th", input_shape = (3, 32, 32)))
	model.add(MaxPooling2D((2, 2),    dim_ordering = "th")) 
	model.add(Convolution2D(50, 3, 3, dim_ordering = "th"))
	model.add(MaxPooling2D((2, 2),    dim_ordering = "th")) 
	model.add(Flatten())
	#model.add(Dense(output_dim = 200))#, W_regularizer = l2(0.01)))
	#model.add(Activation('sigmoid'))
	#model.add(Dropout(0.01))
	model.add(Dense(output_dim = 700, W_regularizer = l2(0.03)))
	model.add(Activation('sigmoid'))
	model.add(Dropout(dropout))
	model.add(Dense(output_dim = 10))
	model.add(Activation('softmax'))
	model.summary()

def divide(x_label, y_label, n):
	x_label, y_label = shuffle_in_unison(x_label, y_label)
	x_valid = x_label[:n, :]
	x_train = x_label[n:, :]
	y_valid = y_label[:n, :]
	y_train = y_label[n:, :]
	print "Divide(Shape) = ", x_train.shape, x_valid.shape, y_valid.shape
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

def label_data(model, x_u, thre, begin, end):
	result = model.predict(x_u)
	print "Finished Predicting Unlabeled Data." 
	'''for i in range(20):
		for j in range(10):
			print "%.3f " %result[i*10][j]
		print ""'''
	print "Threshold Of Entropy = ", thre
	
	newId = []
	newLabel = []
	count = 0
	for i in np.arange(begin, end):
		if entropy(result[i]) < thre:
			maxP = 0.0
			maxId = -1
			for j in range(10):
				if result[i][j] > maxP:
					maxP = result[i][j]
					maxId = j
			newId.append(i)
			newLabel.append(maxId)
			count += 1

	x_new = np.empty([count, 3, 32, 32], dtype = int)
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

x_label, y_label = load_label_data()

model = Sequential()
construct_model(model)
#earlyStopping = EarlyStopping(monitor = 'val_loss', min_delta = 0, patience = 0, verbose = 0, mode = 'auto')
model.summary()
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
#x_label, x_valid, y_label, y_valid = divide(x_label, y_label, 500)
model.fit(x_label, y_label, batch_size = 500, nb_epoch = 50, shuffle = True)
#cv_error(model, x_valid, y_valid)50

#test(model)
x_u = load_unlabel_data()
thre = [0.2, 0.2, 0.2]
drop = [0.25, 0.25, 0.25]

for i in range(2):
	if i==0: x_new, y_new = label_data(model, x_u, thre[i], 0, 45000)
	else: x_new, y_new = label_data(model2, x_u, thre[i], 0, 45000)
	x_train, y_train = merge(x_label, y_label, x_new, y_new)
	
	model2 = Sequential()
	construct_model2(model2, drop[i])
	model2.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
	model2.fit(x_train, y_train, batch_size = y_train.shape[0]/10, nb_epoch = 200, shuffle = True)
	
#pickle.dump(model, open(sys.argv[2], "wb"))
model2.save(sys.argv[2])
#test(model)

'''thre = [0.1, 0.1, 0.1]
drop = [0.2, 0.2, 0.2]
for i in range(3):
	if i==0: x_new, y_new = label_data(model, x_u, thre[i], 0, 45000)
	else: x_new, y_new = label_data(model2, x_u, thre[i], 0, 45000)
	x_train, y_train = merge(x_label, y_label, x_new, y_new)

	model2 = Sequential()
	construct_model2(model2, drop[i])
	model2.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
	model2.fit(x_train, y_train, batch_size = y_train.shape[0]/10, nb_epoch = 1, shuffle = True)
	#cv_error(model2, x_valid, y_valid)
	test(model2, i+3)'''
