from numpy import genfromtxt, array
from math import *
from random import random
import numpy as np
import sys
import pickle

class Spam_Classifier(object):
	def __init__(self):
		# constant
		self.C = 30
		self.slope = 0.28

		self.eta = 30E-7
		self.iterator = 1000
		self.num_feature = 57;
		self.exponential_norm = 100; # avoid sigmoid(z) become 1.0
		self.num_lost_output = 20;
		
		# container
		self.norm = []
		'''self.test_w = [-0.7042, -0.0246, -0.0033, -0.0336, 0.0070, 0.0467, 0.1612, 0.0324, 0.0612, 0.0025, 
		-0.0008, -0.0310, -0.1441, -0.0216, -0.0182, 0.0004, 0.1040, 0.1016, -0.0031, -0.0693, 
		0.1230, 0.1021, 0.1889, 0.3094, 0.0298, -0.5065, -0.1198, -0.2697, 0.0216, -0.1422, 
		-0.0816, -0.2715, -0.2220, -0.0314, 0.2937, -0.1645, 0.0136, -0.0054, 0.2293, 0.0023, 
		-0.0146, 0.2485, -0.3350, 0.1436, -0.0789, -0.1570, -0.2273, 0.2792, 0.1916, -0.0133, 
		-0.1759, -0.0188, 0.0676, 0.1774, 0.0177, -0.1941, 1.8651, -0.1464,] 
		'''
		# other
		self.submit = 1 # 1 for output submit only, 0 for output all

	def resize_raw_data(self):
		# import data from csv by numpy
		raw_data = genfromtxt(sys.argv[1], delimiter = ',')
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
		
	def set_initial_weight(self):
		self.w = np.empty((self.num_feature+1, 1), float);
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

		self.x = self.x/self.norm

	def cost1(self, z):
		if z >= 1: return 0
		else:	   return -self.slope*(z-1)
		
	def cost0(self, z):
		if z <= -1: return 0
		else:       return self.slope*(z+1)

	def grad_cost1(self, z):
		if z >= 1: return 0
		else:	   return -self.slope
		
	def grad_cost0(self, z):
		if z <= -1: return 0
		else:       return self.slope

	def lost(self, z, t):
		lost = 0

		for i in range(self.y.shape[0]):
			lost += self.y[i][0]*self.cost1(z[i][0])+(1-self.y[i][0])*self.cost0(z[i][0])
		
		lost = lost + sum((self.w[1:, :])**2)/2/self.C
		
		print "lost = ", round(lost, 1), t

	def train_model(self):
		i_matrix = np.eye(self.w.shape[0])  
		i_matrix[0][0] = 0
		m = np.empty([self.w.shape[0], self.w.shape[1]], dtype = float)
		v = np.empty([self.w.shape[0], self.w.shape[1]], dtype = float)
		beta1 = 0.9
		beta2 = 0.999
		grad = np.empty([self.w.shape[0], self.w.shape[1]], dtype = float)
		
		for t in range(self.iterator):
			z = np.dot(self.x, self.w)
			if t%(self.iterator/self.num_lost_output) == 0: 
				self.lost(z, t)

			for i in range(self.w.shape[0]):
				grad[i][0] = 0
				for j in range(self.y.shape[0]):
					if self.y[j][0] == 1:
						grad[i][0] += self.grad_cost1(z[j][0])*self.x[j][i]
					else: 
						grad[i][0] += self.grad_cost0(z[j][0])*self.x[j][i]
					if i!=0: grad[i][0] += (self.w[i][0]/self.C)
				
			m = beta1*m + array((1-beta1)* grad    ).reshape(self.w.shape[0], 1)
			v = beta2*v + array((1-beta2)*(grad**2)).reshape(self.w.shape[0], 1)
			m_dot = m/(1-beta1**(t+1))
			v_dot = v/(1-beta2**(t+1))	
			self.w = self.w - self.eta*m_dot/(v_dot**(1/2) + 1E-8)
			
		'''for i in range(self.w.shape[0]):
			print "%.4f," % self.w[i][0],
			if i%10 == 9: print ""
		print ""'''

	def sigmoid(self, num):
		return 1/(1+np.exp(-num/self.exponential_norm))

	def compute_validation_error(self):
		error = 0
		my_error = 0
		self.vali_x = self.vali_x/self.norm
		
		for i in range(self.vali_y.shape[0]):
			y_dot = self.sigmoid(np.dot(self.vali_x[i], self.w))
			if self.vali_y[i][0] != y_dot:
				cross = -(self.vali_y[i][0] * np.log(y_dot) + (1-self.vali_y[i][0]) * np.log(1-y_dot))
				error += cross[0]
			cross = -(self.vali_y[i][0] * np.log(y_dot) + (1-self.vali_y[i][0]) * np.log(1-y_dot))

			if y_dot >= 0.5:
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
		print "Test: my_error = ", round(my_error/self.x.shape[0], 4)



model = Spam_Classifier()
model.resize_raw_data()
model.get_train_data()
model.normalization()
model.set_initial_weight()

model.train_model()
if model.submit == 0: model.compute_validation_error()

#model.compute_test_error()
pickle.dump(model, open(sys.argv[2], "wb")) 

