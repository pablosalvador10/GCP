from fileinput import filename
import os
import pandas as pd
from sklearn import preprocessing
from sklearn import metrics
import pickle

#test
def Inference_job(filename, X_eval,Y_eval):

    """Inference Job

    Returns:
        evaluations
    """

    #Load model 
    model = pickle.load(open(filename,'rb'))
    
    pred_test = model.predict(X_eval)
       
    print("Accuracy on test set : ",model.score(X_eval,Y_eval))
    print("Recall on test set : ",metrics.recall_score(Y_eval,pred_test))
    print("Precision on test set : ",metrics.precision_score(Y_eval,pred_test))
    
    return 


