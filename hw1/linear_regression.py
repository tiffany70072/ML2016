from numpy import genfromtxt, array
from math import *
from random import random
import numpy as np
import pylab as plt

#from pandas import DataFrame, read_csv
#import pandas as pd
#import sys

'''import numpy as np
A = np.array([1,2,3,4,5,6])
B = vec2matrix(A,ncol=2)
A = np.array([0,1,2]).reshape((3L, 1L))
grad threshold
   '''

class Predict_PM25(object):
	def __init__(self):
		
		# constant
		self.lam = 0.03
		self.lam = 1
		self.eta = 3E-6
		self.iterator = 5000
		self.total_id = 18

		self.num_PM25 = 9
		self.num_other = 4

		#self.feature = [-7, -6, -5, -3, -2, -1, 2, 5, 7, 8] # 10, n1
		#self.feature = [-9, -7, -6, -4, -3, -2, -1, 2, 3, 4, 5] # 11, n2, from excel
		#self.feature = [-7, -2, -1, 2, 3, 7, 8] # n3, by plot
		self.feature = [-9, -7, -6, -4, -3, -2, -1, 2, 3, 4, 7, 8] # try + n2
		
		
		# container
		self.x = np.empty((0, 0), float)# 1 dim for PM2.5, change later
		self.y = []
		self.w = []
		self.norm = []

		self.plt_x = []
		self.plt_y = []
		self.record = []
		
		# other
		self.submit = 1 # 1 for output submit only, 0 for output all
	
	def output_array(self, arr):
		fout = open('out.txt','w')
		for i in range(len(arr)):
			for j in range(len(arr[i])): fout.write(str(arr[i][j]) + " ")
			fout.write("\n")
		fout.close()	

	def wind_direc_1(self, degree): # 300
		return cos(degree)
		'''if    degree >=300: return 1
		elif  degree >=330: return 2
		elif  degree < 30:  return 3
		elif  degree < 60:  return 4
		elif  degree < 90:  return 5
		elif  degree < 120: return 6
		elif  degree < 150: return 7
		elif  degree < 180: return 8
		elif  degree < 210: return 9
		elif  degree < 240: return 10
		elif  degree < 270: return 11
		else: return 12'''
	
	def wind_direc_2(self, degree): # 135
		if    degree >= 345 and degree < 15: return 8
		elif  degree < 45:  return 9
		elif  degree < 75:  return 10
		elif  degree < 105: return 11
		elif  degree < 135: return 12
		elif  degree < 165: return 1
		elif  degree < 195: return 2
		elif  degree < 225: return 3
		elif  degree < 255: return 4
		elif  degree < 285: return 5
		elif  degree < 315: return 6
		else: return 7

	def rainfall(self, rain):
		if str(rain) != "nan": return rain
		else: return -1

	def resize_raw_data(self):
		# import data from csv by numpy
		raw_data = genfromtxt('data/train.csv', delimiter = ',')
		if self.submit == 0: print "raw_data.shape = ", raw_data.shape

		raw_data = np.delete(raw_data, 0, axis = 0) # remove top 1 row
		for i in range(3): 
			raw_data = np.delete(raw_data, 0, axis = 1) # remove left 3 column
		
		# reshape data
		self.data = raw_data[0:18, :]
		for i in range(raw_data.shape[0]/self.total_id - 1):
			temp = raw_data[18*(i+1):18*(i+2), :]
			self.data = np.concatenate((self.data, temp), axis = 1)
		if self.submit == 0: print "data.shape = ", self.data.shape # = 18*5760

	def get_train_data(self):
		# total = 5652 = 12*(20*24-9)
		data = self.data # must be faster
		flag = 0
		for month in range(12):
			for hr in range(471): # 24*20-9
				col = 480*month + hr
				temp_x = [[1]]
				temp_y = [[]]
				temp_y[0].append(data[9][col+9]) # real answer
				#self.plt_y.append(data[9][col+9])
		
				for j in range(self.num_PM25): # normal = 9
					temp_x[0].append(data[9][col+8-j])
				
				for j in range(len(self.feature)):
					for k in range(self.num_other):
						temp_x[0].append(data[9+self.feature[j]][col+8-k])

				#temp_x[0].append(self.wind_direc_1(data[9+5][col+8])) # wind
				#temp_x[0].append(self.wind_direc_2(data[9+5][col+8]))
				#temp_x[0].append(data[9+5][col+8]) # wind
				#temp_x[0].append(self.rainfall(data[9+1][col+8])) # rain

				guess = 2*data[9][col+8]-data[9][col+7]
				temp_x[0].append(guess)
				guess = 3*data[9][col+8]-3*data[9][col+7]+data[9][col+6]
			 	temp_x[0].append(guess)
			 	#self.plt_x.append(guess)

				if flag == 0:
					self.x = array(temp_x)
					self.y = array(temp_y)
					flag = 1
				elif flag == 1 and self.submit == 0 and hr>350:
					self.vali_x = array(temp_x)
					self.vali_y = array(temp_y)
					flag = 2
				elif self.submit == 0 and hr>350: #(hr%4 == 1):
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

		#plt.plot(self.plt_x, self.plt_y, 'o')
		#plt.show()	

	def normalization(self):
		self.norm.append(1)
		ave = 20
		for i in range(9): self.norm.append(ave)
		for i in np.arange(10, self.x.shape[1]):
			total = 0
			for j in np.arange(1, 4001, 200): # from the middle, 20 times
				total += self.x[j][i]
			if int(total/20) != 0:
				self.norm.append(int(total/20))
			else: self.norm.append(1)
		
		self.x = self.x/self.norm
		
	def set_initial_weight(self):
		for i in range(self.x.shape[1]): # 10 dim, first is bias(*1), others are PM2.5 from 1 to 9
			self.w.append([random()]) # change float
		self.w = array(self.w)
		if self.submit == 0: print "w.shape = ", self.w.shape

	def lost_function(self, i):
		lost = sum((self.y - np.dot(self.x, self.w))**2) + self.lam*sum(self.w[1:-1]**2)
		
		if i>1*self.iterator/10: self.record.append(lost)
		if (i%(self.iterator/20) == 0 or i==0) and self.submit == 0: 
			print "lost = ", round(lost, 1), i

	def train_model(self):
		i_matrix = np.eye(self.w.shape[0])  
		i_matrix[0][0] = 0
		group_num = (self.x.shape[0]/100) - 1

		# initialize for adam grad
		m = np.empty([self.w.shape[0], self.w.shape[1]], dtype = float)
		v = np.empty([self.w.shape[0], self.w.shape[1]], dtype = float)
		t = 0
		eta_adam = 0.0000003 # 0.000001 is too big
		beta1 = 0.9
		beta2 = 0.999

		# use np.dot() to multiply i dim matrix(vector)
		for t in range(self.iterator):
			if t%(self.iterator/20) == 0 or t==0:
				self.lost_function(t) 

			# stochastic
			if t<self.iterator/2:
				for j in range(100):
					sta = group_num*(j%100)
					if j!= 99: 
						fin = group_num*(j%100+1)
					else: # the last one
						fin = self.x.shape[0]
					gra = 2 * (sum(-(self.y[sta:fin, :] - np.dot(self.x[sta:fin, :], self.w))*self.x[sta:fin, :])
						+ self.lam * sum(np.dot(i_matrix, self.w)))
					for k in range(len(self.w)): # change later
						self.w[k][0] -= (gra*self.eta)[k]
			
			# adam grad
			else:
				gra = 2 * (sum(-self.x * (self.y - np.dot(self.x, self.w))) 
					+ self.lam * sum(np.dot(i_matrix, self.w)))
				
				m = beta1*m + array((1-beta1)* gra    ).reshape(self.w.shape[0], 1)
				v = beta2*v + array((1-beta2)*(gra**2)).reshape(self.w.shape[0], 1)
				m_dot = m/(1-beta1**t)
				v_dot = v/(1-beta2**t)
				
				self.w = self.w - eta_adam*m_dot/(v_dot**(1/2) + 1E-8)
			
			#for j in range(len(self.w)): self.w[j][0] -= (gra*self.eta)[j]
		if self.submit == 0:
			plt.plot(self.record)
			plt.show()	
		#for i in range(len(self.w)): print round(self.w[i][0], 1),
		#print "\n"

	def compute_validation_error(self):
		total = 0
		my_error = 0
		self.vali_x = self.vali_x/self.norm
		
		for i in range(self.vali_y.shape[0]):
			total += (self.vali_y[i] - np.dot(self.vali_x[i], self.w))**2	
			my_error += (abs(self.vali_y[i] - np.dot(self.vali_x[i], self.w)))
		#print "total = ", total
		#print "my_error = ", 
		print my_error/self.vali_y.shape[0]

	def compute_test_error(self):	
		print "id,value"
		test_data = genfromtxt('data/test_X.csv', delimiter = ',')
		for i in range(len(test_data)/self.total_id):	
			row = i*self.total_id
			test_x = [] # change to clear later
			test_x.append([1])

			# input pm2.5 1~9 
			for j in range(self.num_PM25): test_x[0].append(test_data[row+9][10-j])
			# input other 10 features
			for j in range(len(self.feature)): 
				for k in range(self.num_other):
					test_x[0].append(test_data[row+9+self.feature[j]][10-k])
			
			#test_x[0].append(self.wind_direc_1(test_data[row+9+5][10]))
			#test_x[0].append(self.wind_direc_2(test_data[row+9+5][10]))
			#test_x[0].append(test_data[row+9+5][10])
			#test_x[0].append(self.rainfall(test_data[row+9+1][10])) # rain
			
			guess = 2*test_data[row+9][10]-test_data[row+9][9]
			test_x[0].append(guess)
			guess = 3*test_data[row+9][10]-3*test_data[row+9][9]+test_data[row+9][8]
			test_x[0].append(guess)
			
			test_x = array(test_x)
			test_x = test_x/self.norm
				
			#print "id_%d,%d" % (i, int(np.dot(test_x[0], self.w)[0]))
			print "id_%d,%.3f" % (i, float(np.dot(test_x[0], self.w)[0]))


model = Predict_PM25()
model.resize_raw_data()
model.get_train_data()
model.normalization()
model.set_initial_weight()
#w_init = model.w
#model.vali_x = model.vali_x/model.norm
#lam_list = [0.01, 0.03, 0.1, 0.2, 0.3, 0.6, 0.8, 1, 3, 10, 30, 100]
#for i in range(len(lam_list)):
#	model.lam = lam_list[i]
#	print ""
#	print model.lam,
#	model.w = w_init
model.train_model()
if model.submit == 0: 
	model.compute_validation_error()
else: 
	model.compute_test_error()
