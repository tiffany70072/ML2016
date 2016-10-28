from numpy import genfromtxt, array
from math import *
from random import random
import numpy as np
import pylab as plt
import sys
import pickle

class Spam_Classifier(object):
	def __init__(self):
		# constant
		self.lam = 0.0001
		self.eta = 1000E-6
		self.iterator = 20000
		self.num_feature = 57
		self.exponential_norm = 100 # avoid sigmoid(z) become 1.0
		self.num_lost_output = 5
		
		# container
		self.norm = []

		# specific value
		self.w = []
		'''self.test_w = [-0.145, 0.426, -0.034, -0.425, 0.308, 0.417, 0.373, -0.393, 0.236, -0.395, 
		-0.145, 0.119, 0.321, 0.400, -0.034, 0.196, 0.523, -0.130, 0.445, 0.253, 
		0.377, 0.432, 0.281, 0.386, 0.183, -0.514, 0.287, 0.115, 0.272, -0.188, 
		-0.089, -0.214, -0.325, 0.255, 0.457, -0.154, -0.288, -0.296, 0.363, 0.124, 
		0.476, 0.422, -0.484, 0.267, 0.031, 0.109, 0.052, 0.436, 0.384, 0.450, 
		-0.033, -0.503, -0.437, -0.405, -0.375, 0.368, 3.319, 0.132]		
		'''
		# other
		self.submit = 1 # 1 for output submit only, 0 for output all
	
	def output_weight(self):
		filename = str(sys.argv[2])
		fout = open(filename, 'w')
		for i in range(self.w.shape[0]):
			fout.write(str(self.w[i][0]) + "\n")
		fout.close()	

	def resize_raw_data(self):
		# import data from csv by numpy
		filename = str(sys.argv[1])
		raw_data = genfromtxt(filename, delimiter = ',')
		if self.submit == 0: print "raw_data.shape = ", raw_data.shape

		# reshape data
		raw_data = np.delete(raw_data, 0, axis = 1) # remove left 3 column
		self.data = array(raw_data)
		if self.submit == 0: print "data.shape = ", self.data.shape # = 4000*59

	def get_train_data(self):
		for i in range(4001): # total num of data
			temp_x = [[1]]
			temp_y = [[self.data[i][59-2]]] # label = the last element in one row
			
			for j in range(self.num_feature): 
				temp_x[0].append(self.data[i][j]);

			if i == 0:
				self.x = array(temp_x)
				self.y = array(temp_y)	
			elif i == 1 and self.submit == 0:
				self.vali_x = array(temp_x)
				self.vali_y = array(temp_y)
			elif self.submit == 0 and i%4 == 1:
				self.vali_x = np.concatenate((self.vali_x, array(temp_x)), axis = 0)
				self.vali_y = np.concatenate((self.vali_y, array(temp_y)), axis = 0)		
			else:
				self.x = np.concatenate((self.x, array(temp_x)), axis = 0)
				self.y = np.concatenate((self.y, array(temp_y)), axis = 0)
			
		if self.submit == 0:
			print "x.shape = ", self.x.shape
			print "y.shape = ", self.y.shape
			print "vali_x.shape = ", self.vali_x.shape
			print "vali_y.shape = ", self.vali_y.shape
		
	def set_initial_weight(self):
		self.w = np.empty((self.x.shape[1], 1), float);
		for i in range(self.w.shape[0]): # Dim = 57
			self.w[i][0] = random() - 0.5 
			#self.w[i][0] = self.test_w[i]

	def normalization(self):
		self.norm = [1.0, 0.1, 0.028, 0.281, 0.005, 0.158, 0.275, 0.018, 0.13, 0.11, 
		0.273, 0.072, 0.573, 0.114, 0.111, 0.013, 0.142, 0.155, 0.126, 1.536, 
		0.106, 0.874, 1.0, 0.182, 0.117, 0.399, 0.233, 0.612, 0.045, 1.0, 
		0.045, 0.045, 1.0, 1.0, 1.0, 0.045, 0.06, 0.011, 1.0, 1.0, 0.015, 
		1.0, 0.093, 1.0, 1.0, 0.41, 0.09, 1.0, 1.0, 0.012, 0.11, 
		0.007, 0.439, 0.052, 0.022, 3.253, 51.6, 325.85]

		'''self.norm.append(1)
		for i in range(self.num_feature):
			total = 0
			for j in np.arange(1, 3001, 150): # from the middle, 20 times
				total += self.x[j][i+1]
			if total/20 != 0:
				self.norm.append((total/20))
			else: self.norm.append(1)
		
		for i in range(len(self.norm)):
			print round(self.norm[i], 3),'''

		self.x = self.x/self.norm

	def sigmoid(self, num):
		return 1/(1+np.exp(-num/self.exponential_norm))

	def lost(self, y_dot, t):
		lost = 0
		for i in range(self.y.shape[0]):
			if self.y[i][0] != y_dot[i][0]:
				cross = -(self.y[i][0] * np.log(y_dot[i][0]) + (1-self.y[i][0]) * np.log(1-y_dot[i][0]))
				lost += cross

		print "lost = ", round(lost, 1), t

	def train_model(self):
		i_matrix = np.eye(self.w.shape[0])  
		i_matrix[0][0] = 0
		m = np.empty([self.w.shape[0], self.w.shape[1]], dtype = float)
		v = np.empty([self.w.shape[0], self.w.shape[1]], dtype = float)
		beta1 = 0.9
		beta2 = 0.999

		for t in range(self.iterator):
			y_dot = self.sigmoid(np.dot(self.x, self.w))
			if t%(self.iterator/self.num_lost_output) == 0: 
				self.lost(y_dot, t)

			# adam grad
			gra = -sum((self.y - y_dot)*self.x) + 2*self.lam * sum(np.dot(i_matrix, self.w))
			m = beta1*m + array((1-beta1)* gra    ).reshape(self.w.shape[0], 1)
			v = beta2*v + array((1-beta2)*(gra**2)).reshape(self.w.shape[0], 1)
			m_dot = m/(1-beta1**(t+1))
			v_dot = v/(1-beta2**(t+1))	
			self.w = self.w - self.eta*m_dot/(v_dot**(1/2) + 1E-8)

	def compute_validation_error(self):
		error = 0
		my_error = 0
		#self.vali_x = self.vali_x/self.norm
		for i in range(self.vali_y.shape[0]):
			y_dot = self.sigmoid(np.dot(self.vali_x[i], self.w))
			if self.vali_y[i][0] != y_dot:
				cross = -(self.vali_y[i][0] * np.log(y_dot) + (1-self.vali_y[i][0]) * np.log(1-y_dot))
				error += cross[0]

			if y_dot[0] >= 0.5:
				my_error += 1 - self.vali_y[i][0]
			else:
				my_error += self.vali_y[i][0]

		print "error = ", round(error/self.vali_y.shape[0], 4)
		print "my_error = ", my_error/self.vali_y.shape[0]

		my_error = 0
		for i in range(self.x.shape[0]):
			y_dot = self.sigmoid(np.dot(self.x[i], self.w))

			if y_dot >= 0.5:
				my_error += 1 - self.y[i][0]
			else:
				my_error += self.y[i][0]

		#print "Test: error = ", round(error/self.x.shape[0], 4),
		print "Test: my_error = ", round(my_error/self.x.shape[0], 4)

	def compute_test_error(self):	
		print "\nid,label"
		test_data = genfromtxt('spam_data/spam_test.csv', delimiter = ',')
		
		for i in range(len(test_data)):	
			test_x = [] # change to clear later
			test_x.append([1])
			for j in range(self.num_feature): 
				test_x[0].append(test_data[i][j+1])
			
			test_x = array(test_x)
			test_x = test_x/self.norm
			y_dot = self.sigmoid(np.dot(test_x, self.w))
			
			if y_dot >= 0.5:
				result = 1
			else:
				result = 0
			print "%d,%d" % (i+1, result)


model = Spam_Classifier()
model.resize_raw_data()
model.get_train_data()
model.normalization()

#model.vali_x = model.vali_x/model.norm
lam = [0.0001]#, 0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1]
for i in range(len(lam)):
	print "lam = ", lam[i]
	model.set_initial_weight()
	model.lam = lam[i]
	model.train_model()
	if model.submit == 0: 
		model.compute_validation_error()
#else:
pickle.dump(model, open(sys.argv[2], "wb")) 
#model.output_weight()
#model.compute_test_error()
