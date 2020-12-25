# Team Project
# usage:  
#
# 
# History
# -    -    -    -    -    -    -    -    -    -    -    -    -    -    -
# Name		Date		Description
# scl     	11/07/2020 	Initial framework implemented with syntethic data
# 						X = features selected, y = classification
# scl 		11/08/2020	Added implementation for train and split
# scl		11/08/2020	Add implementation for configuration file
# Aashish   11/14/2020  Add tests for nflDataScience
# Aashish   11/16/2020  Add K Means implementation
# Aashish   11/22/2020  Refactor precision, clean up main
# -    -    -    -    -    -    -    -    -    -    -    -    -    -    -
from team_utility import dataScience, getConfigs, getConfigsJobs
from team_ml import NNObj, NBObj, SVMObj, KMObj

jobs, PRECISION = getConfigsJobs()
for j in range(jobs):
	filename,feature_col1,feature_col2,classifi_col,k = getConfigs('job_'+str(j)) 

	# Start the DataScience Flow
	# Aquire, Clean & Structure the data
	data = dataScience(filename)
	data.featureSelection(feature_col1, feature_col2, classifi_col)

	svm = SVMObj(data.X_train,data.y_train)
	nbz = NBObj(data.X_train,data.y_train)
	nbr = NNObj(data.X_train,data.y_train,k)
	kmm = KMObj(data.X_train, data.y_train, k)

	# Replace with the test and training split
	test = data.X_test

	y_predict_nbr = nbr.makePredictiction(data.X_test)
	y_predict_nbz = nbz.makePredictiction(data.X_test)
	y_predict_svm = svm.makePredictiction(data.X_test)
	y_predict_kmm = kmm.makePredictiction(data.X_test)

	print('\n\n - - Job ', j, ' Execution - - ')
	print('Accuracy KNN', round(nbr.getAccuracyScore(data.y_test, y_predict_nbr),PRECISION))
	print('Accuracy NB ', round(nbz.getAccuracyScore(data.y_test, y_predict_nbz),PRECISION))
	print('Accuracy SVM', round(svm.getAccuracyScore(data.y_test, y_predict_svm),PRECISION))
	print('Accuracy KMM', round(kmm.getAccuracyScore(data.y_test, y_predict_kmm),PRECISION))
	print('F1 KNN', round(nbr.getF1Score(data.y_test, y_predict_nbr),PRECISION))
	print('F1 NB ', round(nbz.getF1Score(data.y_test, y_predict_nbz),PRECISION))
	print('F1 SVM', round(svm.getF1Score(data.y_test, y_predict_svm),PRECISION))
	print('F1 KMM', round(kmm.getF1Score(data.y_test, y_predict_kmm),PRECISION))


