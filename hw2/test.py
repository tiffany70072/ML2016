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
		self.num_feature = 57
		self.exponential_norm = 100 

		# container
		self.norm = []

		
	def set_weight(self):
		modelname = str(sys.argv[1])
		weight = genfromtxt(modelname, delimiter = '\n')
		self.w = np.empty((self.num_feature+1, 1), float);
		#print weight
		for i in range(self.w.shape[0]): 
			self.w[i][0] = weight[i]

	def normalization(self):
		self.norm = [1.0, 0.1, 0.028, 0.281, 0.005, 0.158, 0.275, 0.018, 0.13, 0.11, 
		0.273, 0.072, 0.573, 0.114, 0.111, 0.013, 0.142, 0.155, 0.126, 1.536, 
		0.106, 0.874, 1.0, 0.182, 0.117, 0.399, 0.233, 0.612, 0.045, 1.0, 
		0.045, 0.045, 1.0, 1.0, 1.0, 0.045, 0.06, 0.011, 1.0, 1.0, 0.015, 
		1.0, 0.093, 1.0, 1.0, 0.41, 0.09, 1.0, 1.0, 0.012, 0.11, 
		0.007, 0.439, 0.052, 0.022, 3.253, 51.6, 325.85]

	def sigmoid(self, num):
		return 1/(1+np.exp(-num/self.exponential_norm))

	def compute_test_error(self):	
		filename = str(sys.argv[2])
		test_data = genfromtxt(filename, delimiter = ',')
		
		foutname = str(sys.argv[3])
		fout = open(foutname, 'w')
		fout.write("id,label\n")
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
			#print "%d,%d" % (i+1, result)
			fout.write("%d,%d\n" % (i+1, result))
		
		fout.close()	


#model = Spam_Classifier()
model = pickle.load(open(sys.argv[1], "rb"))
#model.normalization()

#model.set_weight()
model.compute_test_error()
