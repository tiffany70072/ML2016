import sys
import pickle
import random # shuffle
import string # punctuation
import numpy as np
import matplotlib.pyplot as plt
from math import isnan
from string import digits # remove digit
from stop_words import get_stop_words
from wordsegment import segment
from compiler.ast import flatten
from gensim.models import Word2Vec
from sklearn.cluster import KMeans, AffinityPropagation, AgglomerativeClustering
from scipy.spatial.distance import cosine # cosine similarity, not distance
#import codecs

class Model(object):
	def __init__(self):
		self.iter_kMeans = 300 	# default = 300
		self.stopwords = get_stop_words('english')
		self.myStopwords = self.get_my_stop_words()
		self.dimWordVector = 300 # default = 300		#self.doc = []  
		
	def get_my_stop_words(self):
		f_stopwords = open("stop_words.txt", "r") 
		self.myStopwords = []  
		for line in f_stopwords: self.myStopwords.append(line[:-1])	
		f_stopwords.close()
		return self.myStopwords

	def preprocessSentence(self, l): # can adjust to be faster
		l = l.replace("-", " ")
		l = l.replace("_", " ")
		l = l.replace("/", " ")
		l = l.replace(".", " ")
		l = l.replace(":", " ")
		l = l.translate(string.maketrans("",""), string.punctuation) # remove punctuation
		l = l.lower().translate(None, digits)
		l = [w for w in l.split() if (len(w) <= 24 and w[0:3] != "http")] # 20?
		l = flatten([segment(w) for w in l])
		l = [w for w in l if w not in self.stopwords] # remove stopwords
		l = [w for w in l if w not in self.myStopwords]
		return l

	def load(self, filename):
		#f = codecs.open(filename, 'r', encoding = 'utf8')
		f = open(filename, "r") 
		i = 0
		for line in f:  
			self.doc.append(self.preprocessSentence(line))
			if (i%2000) == 0: print i
			i += 1
		f.close()

	def preprocessData(self, run_again):
		if run_again == 1:
			self.doc = []
			#self.doc = pickle.load(open("model-doc5", "rb"))
			model.load(sys.argv[1] + "/title_StackOverflow.txt")
			model.load(sys.argv[1] + "/docs.txt")
			for i in range(5):
				model.load(sys.argv[1] + "/title_StackOverflow.txt")
			pickle.dump(self.doc, open("model-doc6", "wb"))
		else:
			self.doc = pickle.load(open("model-doc6", "rb"))
		#self.doc[2043] = ['excel'] # command later
		print "doc.len = ", len(self.doc) 

	def wordVector(self, run_again):
		print "Word to Vector"
		if run_again == 1:
			self.wordModel = Word2Vec(self.doc, size = self.dimWordVector, window = 10, min_count = 6, workers = 2, iter = 15)
			self.wordModel.init_sims(replace = True)
			#self.wordModel.save("model-wordVector6")
		else:
			self.wordModel = Word2Vec.load("model-wordVector6")  # you can continue training with the loaded model!
		
		#print self.wordModel.similarity('return', 'dataset')
		#print self.wordModel.most_similar_cosmul(positive = ['code', 'work'], negative = ['type'], topn = 1)
		#print wordModel.similarity('mac', 'os')
		#print self.wordModel.similar_by_word('mac', topn = 10, restrict_vocab = None)

	def wordEmbedding(self, run_again):
		print "Word Embedding"
		if run_again == 1:
			self.titleVector = np.zeros([20000, self.dimWordVector])
			self.noVector = []
			for i in range(20000):
				temp = np.zeros([self.dimWordVector])
				count = 0.0
				for j in range(len(self.doc[i])):
					try:
						temp += self.wordModel[self.doc[i][j]]
						count += 1.0
					except KeyError: continue

				if count != 0: self.titleVector[i] = np.array(temp/count)
				else: # if no word of the title in the training data 
					self.noVector.append(i)
					#print i

			print "Num of no word vector = ", len(self.noVector)
			#pickle.dump(self.titleVector, open("model-titleVector6", "wb"))
		else:
			self.titleVector = pickle.load(open("model-titleVector6", "rb"))

	def cluster(self):
		print "KMeans Clustering"
		self.kmeans = KMeans(n_clusters = 20, max_iter = self.iter_kMeans, random_state = 0).fit(self.titleVector)
		#print "Center = ", self.kmeans.cluster_centers_

		self.predict_class = []
		for i in range(20000):
			self.predict_class.append(self.kmeans.predict([self.titleVector[i]]))
		#print self.predict_class[11726]

	def predictBoundary(self, numLeader, numMember):
		print "Calculate Threshold"
		total = 0
		threIndex = numLeader/20
		ranLeader = [n for n in range(numLeader)] # only shuffle one time
		random.shuffle(ranLeader)
		ranMember = [n for n in range(numMember)] # need to shuffle for each leader
		simBoundaryScore = np.empty((numLeader), dtype = float) # store the boundary
		simScore = np.zeros((numMember), dtype = float)
		for i in range(numLeader):
			random.shuffle(ranMember)
			for j in range(numMember):
				cos = cosine(self.titleVector[ranLeader[i]], self.titleVector[ranMember[j]])
				if not isnan(cos): simScore[j] = cos

			simScore = np.sort(simScore)
			#print simScore[threIndex], simScore[threIndex+1]
			simBoundaryScore[i] = (simScore[threIndex])# + simScore[threIndex+1])/2
		simBoundaryScore = np.sort(simBoundaryScore)
		return np.average(simBoundaryScore)
		#return np.average(simBoundaryScore[numLeader*0.1:numLeader*0.9])

	def classifier(self):
		self.threshold = 0.57
		#self.threshold = self.predictBoundary(1800, 1800) 
		print "Threshold = ", self.threshold
		print "Test"
		#exit()
		fout = open(sys.argv[2], 'w')
		fout.write("ID,Ans\n")
		i = 0
		count = 0
		with open(sys.argv[1] + "/check_index.csv") as f:
			next(f)
			for line in f:
				ids = line.split(",")
				cos = cosine(self.titleVector[int(ids[1])], self.titleVector[int(ids[2][:-1])])
				if cos < self.threshold:
					count += 1
					fout.write("%d,%d\n" %(i, 1))
				else: fout.write("%d,%d\n" %(i, 0))

				i += 1
				sys.stdout.write("\r id: %d" % i)
				sys.stdout.flush()
				#if i >= 500000: break
		print "", i
		print "Same cluster = ", count

	def pcaVisualization(self):
		colors = ['b', 'g', 'r', 'c', 'pink', 'yellow', 'grey', 'purple', 'salmon', 'aqua', 
		'navy', 'pink', 'b', 'orangered','gold', 'greenyellow', 'mistyrose', 'c', 'aqua', 'coral']

		print "Visualization shape = ", self.titleVector.shape
		colors_list = []
		n = 2000
		for i in range(n): colors_list.append(colors[self.predict_class[i][0]])

		plt.scatter(self.titleVector[:n, 0], self.titleVector[:n, 1], c = colors_list)#, alpha = 0.8)
		plt.show()
		plt.scatter(self.titleVector[:n, 2], self.titleVector[:n, 3], c = colors_list)#, alpha = 0.8)
		plt.show()

	def translationMean(self):
		mean = np.average(self.titleVector, axis = 0)
		print mean.shape
		self.titleVector -= mean

	def test(self):
		print "Test"
		fout = open(sys.argv[2], 'w')
		fout.write("ID,Ans\n")
		i = 0
		with open(sys.argv[1] + "/check_index.csv") as f:
			next(f)
			for line in f:
				ids = line.split(",")
				c1 = self.predict_class[int(ids[1])]
				c2 = self.predict_class[int(ids[2][:-1])]
				fout.write("%d,%d\n" %(i, c1==c2))

				i += 1
				sys.stdout.write("\r id: %d" % i)
				sys.stdout.flush()
		print "", i

	def agglomerativeClustering(self):
		print "Agglomerative Clustering"
		self.ag = AgglomerativeClustering(n_clusters = 20)
		#print "Center = ", self.kmeans.cluster_centers_
		self.ag.fit(self.titleVector)
		self.predict_class = self.ag.fit_predict(self.titleVector)
		#print self.ag.fit_predict(self.titleVector)[0]
		#print self.ag.label_.shape
		
print ""

model = Model()
model.preprocessData(0) # can command if wordVector and wordEmbedding are all zero
#print model.doc[2043]
#print model.doc[7453]
#for i in range(5): print model.doc[139+i]
#print model.doc[7490]
model.wordVector(0) # get wordModel, set zero -> dim = 300
model.wordEmbedding(1) # get titleVector, set zero -> dim = 300
#model.translationMean()
model.cluster()
#model.agglomerativeClustering()
#model.classifier() # don't need test(), different method
#model.pcaVisualization()
#exit()
model.test()
