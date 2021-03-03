
# Imports for ML API from sklearn
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.cluster import KMeans

from sklearn.metrics import accuracy_score
from sklearn.metrics.cluster import completeness_score
from sklearn.metrics import f1_score
from sklearn.metrics.cluster import homogeneity_score


class NNObj:
    def __init__(self, X, y, k):
    	self.__score = 0
    	self.__algo = ""
    	self.__model = self.getModelParam(k, X, y)

    def getModelParam(self, k, features, classification):
    	self.__k = k
    	x = features 
    	y = classification
    	self.__model = KNeighborsClassifier(n_neighbors=self.__k).fit(x,y)
    	return self.__model

    def makePredictiction(self, data):
    	predicted_classification = self.__model.predict(data)
    	return predicted_classification

    def getAccuracyScore(self, y_true, y_pred):
    	self.__score = accuracy_score(y_true, y_pred)
    	return self.__score
	
    def getF1Score(self, y_true, y_pred):
    	self.__score = f1_score(y_true, y_pred, average = 'macro')
    	return self.__score

class NBObj:
	def __init__(self, X, y):
		self.__score = 0
		self.__model = self.getModelParam(X, y)

	def getModelParam(self, features, classification):
		x = features 
		y = classification
		self.__model = GaussianNB().fit(x,y)
		return self.__model

	def makePredictiction(self, data):
		predicted_classification = self.__model.predict(data)
		return predicted_classification

	def getAccuracyScore(self, y_true, y_pred):
		self.__score = accuracy_score(y_true, y_pred)
		return self.__score

	def getF1Score(self, y_true, y_pred):
		self.__score = f1_score(y_true, y_pred, average = 'macro')
		return self.__score

class SVMObj:
	def __init__(self, X, y):
		self.__score = 0
		self.__model = self.getModelParam(X, y)

	def getModelParam(self, features, classification):
		x = features 
		y = classification
		self.__model = SVC(gamma='auto').fit(x,y)
		return self.__model

	def makePredictiction(self, data):
		predicted_classification = self.__model.predict(data)
		return predicted_classification

	def getAccuracyScore(self, y_true, y_pred):
		self.__score = accuracy_score(y_true, y_pred)
		return self.__score
	
	def getF1Score(self, y_true, y_pred):
		self.__score = f1_score(y_true, y_pred, average = 'macro')
		return self.__score

class KMObj:
    def __init__(self, X, y, k):
    	self.__score = 0
    	self.__model = self.getModelParam(k, X, y)

    def getModelParam(self, k, features, classification):
    	self.__k = k
    	x = features 
    	y = classification
    	self.__model = KMeans(n_clusters = self.__k).fit(x,y)
    	return self.__model

    def makePredictiction(self, data):
    	predicted_classification = self.__model.predict(data)
    	return predicted_classification

    def getAccuracyScore(self, y_true, y_pred): # Use completeness to sub for accuracy
    	self.__score = completeness_score(y_true, y_pred) 
    	return self.__score

    def getF1Score(self, y_true, y_pred): # Use homogeneity to sub for F1
    	self.__score = homogeneity_score(y_true, y_pred) 
    	return self.__score

