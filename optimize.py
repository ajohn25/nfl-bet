# Team Project
# usage:  
#
# 
# History
# -    -    -    -    -    -    -    -    -    -    -    -    -    -    -
# Name		Date		Description
# Aashish   11/29/20    Add functions for optimizing models
# -    -    -    -    -    -    -    -    -    -    -    -    -    -    -
from team_utility_optimize import dataScienceOptimize, getConfigs, getConfigsJobs,getProjectReport
from team_ml import NNObj, NBObj, SVMObj, KMObj

def optModels():
    features = ['Comp', 'Att', 'Pct', 'Yds', 'Int', 'TD'] # Add/Remove TD from this list
    models = ['KNN', 'NB', 'SVM', 'KM']
    filename =  './data/nfl_acquisition.csv'
    classifi_col = 'Class'
    PRECISION = 3
    data = dataScienceOptimize(filename)

    for model in models:
        top_acc = 0
        top_1 = 'A'
        top_2 = 'B'

        for feature1 in features:
            for feature2 in features:
                # if feature1 == feature2: # Use this to disallow the same feature being used twice
                #     break
                feature_col1 = feature1
                feature_col2 = feature2
                data.featureSelection(feature_col1, feature_col2, classifi_col, .859)

                if model == 'SVM': # Simulate dynamic dispatch for object type
                    mObj = SVMObj(data.X_train,data.y_train)
                elif model == 'KNN':
                    mObj = NNObj(data.X_train,data.y_train, 3)
                elif model == 'NB':
                    mObj = NBObj(data.X_train,data.y_train)
                elif model == 'KM':
                    mObj = KMObj(data.X_train,data.y_train, 2)

                y_predict_mObj = mObj.makePredictiction(data.X_test)
                mObj_acc = mObj.getAccuracyScore(data.y_test, y_predict_mObj)
                
                if mObj_acc > top_acc: # Record conditions for top accuracy score
                    top_acc = mObj_acc
                    top_1 = feature1
                    top_2 = feature2
                    top_f1 = mObj.getF1Score(data.y_test, y_predict_mObj)

        top_acc = round(top_acc,PRECISION)
        top_f1 = round(top_f1,PRECISION)
        print(model, "Maximized Accuracy Score:", top_acc, "Features:", top_1, ",", top_2)
        print(model, "F1 Score:", top_f1)


def optK():

    filename =  './data/nfl_acquisition.csv'
    classifi_col = 'Class'
    PRECISION = 3
    data = dataScienceOptimize(filename)
    feature_col1 = 'Yds' # Use the Optimized Features for each model
    feature_col2 = 'Pct'

    top_acc = 0
    top_k = 0
    for k in range(2, 10):
       
        data.featureSelection(feature_col1, feature_col2, classifi_col, 0.478) # Use Optimized Split for each model
        mObj = NNObj(data.X_train,data.y_train, k) # Change model

        y_predict_mObj = mObj.makePredictiction(data.X_test)
        mObj_acc = mObj.getAccuracyScore(data.y_test, y_predict_mObj)
        
        if mObj_acc > top_acc: # Record conditions for top accuracy score
            top_acc = mObj_acc
            top_k = k

    top_acc = round(top_acc,PRECISION)
    print("NN Maximized Accuracy Score:", top_acc, "K:", top_k)

def optSplit():

    filename =  './data/nfl_acquisition.csv'
    classifi_col = 'Class'
    PRECISION = 3
    data = dataScienceOptimize(filename)
    feature_col1 = 'Yds' # Use the Optimized Features for each model
    feature_col2 = 'Pct'

    top_acc = 0
    top_pct = 0

    for val in range(100, 900): # Check from 10 to 90 % training set
       
        data.featureSelection(feature_col1, feature_col2, classifi_col, val/1000)

        mObj = NNObj(data.X_train,data.y_train, 3) # Change model here

        y_predict_mObj = mObj.makePredictiction(data.X_test)
        mObj_acc = mObj.getAccuracyScore(data.y_test, y_predict_mObj)
        
        if mObj_acc > top_acc: # Record conditions for top accuracy score
            top_acc = mObj_acc
            top_pct = val/1000

    top_acc = round(top_acc,PRECISION)
    print("KNN Maximized Accuracy Score:", top_acc, "Percent:", top_pct)

def main(): # Select which optimization to run
     optModels()
  #  optK()
  #  optSplit()
  
main()
