# -*- coding: utf-8 -*-
"""
Created on Wed Jul 18 08:51:47 2018

@author: Narayanan Abishek
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import explained_variance_score
 
lamda = 0.01

#original dataset
def get_data():
    airquality_df = pd.read_csv('D:\Projects\Kaggle\Datasets\AirQuality UCI\AirQualityUCI.csv',sep=';',decimal=',')
    airquality_df = airquality_df.fillna(value = 0, axis = 1)  #filling null with zeros 
    airquality_df = airquality_df.drop(['Time'],axis = 1)

    return airquality_df
    
    
#feature scaling for the gradient descent to converge faster
def feature_scaling(x):
    
    X_mean = np.mean(x,axis=0)
    X_sigma = np.std(x,axis=0)
    X = np.divide((x - X_mean),X_sigma)
    return X

#initializing the weights and bias
def initialize_params_linear():
    
    w = tf.Variable(tf.truncated_normal([90, 1], mean=0.0, stddev=1.0, dtype=tf.float64))
    b = tf.Variable(tf.zeros(1, dtype = tf.float64))
    return w,b

def calc(X,Y,w,b):
    
    predictions = tf.add(tf.matmul(X,w),b)
    regularizer = tf.nn.l2_loss(w)
    error = tf.reduce_mean(tf.square(Y-predictions) + (lamda * regularizer) )
    return predictions,error #returns tensor

def linear_regression_simple(train_X,train_Y,val_X,val_Y,test_X,test_Y,w,b,learning_rate = 0.5,num_epochs=7500):
    
    
    predictions,cost = calc(train_X,train_Y,w,b) #yhat and cost
    points = [[],[]] #to store train_set_cost
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost) 
    #initialize the optimizer before the global initialization, since optimizer creates variables leads in memory leak
    init = tf.global_variables_initializer()
    
    with tf.Session() as sess:
        
        
        sess.run(init)
        
        for i in list(range(num_epochs)):
           
            sess.run(optimizer)
            if i%10 == 0:
                points[0].append(i+1) #iteration
                points[1].append(sess.run(cost)) #cost

        pred = predictions.eval() #to convert predictions <tensor> into pred <numpy array>
        
        valid_predictions,valid_cost = calc(val_X,val_Y,w,b) #yhat, cost for validation set
    
        val_Y_pred = valid_predictions.eval() #to convert valid_predictions<tensor> into valid_pred<numpy>
                
        plt.figure()
        plt.plot(points[0], points[1], 'r--') #red marking
        plt.axis = (0,num_epochs,0,20)
        plt.show()
        
        test_predictions,test_cost = calc(test_X,test_Y,w,b)
        test_Y_pred = test_predictions.eval()
        
    from sklearn import metrics 
   
    print("\nTraining set accuracy: \n")
    print('Mean Absolute Error:', metrics.mean_absolute_error(train_Y,pred))  
    print('Mean Squared Error:', metrics.mean_squared_error(train_Y,pred))  
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(train_Y,pred)))
        
    print("\nValidation set accuracy: \n")
    print('Mean Absolute Error:', metrics.mean_absolute_error(val_Y, val_Y_pred))  
    print('Mean Squared Error:', metrics.mean_squared_error(val_Y, val_Y_pred))  
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(val_Y, val_Y_pred)))
        
    print("\nTesting set accuracy: \n")
    
   
    
    print('Mean Absolute Error:', metrics.mean_absolute_error(test_Y, test_Y_pred))  
    print('Mean Squared Error:', metrics.mean_squared_error(test_Y, test_Y_pred))  
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(test_Y, test_Y_pred)))

    print("\nActual Train Value: ", train_Y[82], "Predicted Train value: ", pred[82])
    print("\nActual Validation Value: ", val_Y[834], "Predicted Validation value: ", val_Y_pred[834])
    print("\nActual Test Value: ", test_Y[948], "Predicted Test value: ", test_Y_pred[948])
    
def model():
    
    my_df = get_data()
    w,b = initialize_params_linear()
    
    x = my_df.iloc[:,1:13]
    y = my_df.iloc[:,13]
    
    from sklearn.preprocessing import PolynomialFeatures
    
    poly = PolynomialFeatures(degree=2)
    
    my_old_X = poly.fit_transform(x)
    
    my_X = np.nan_to_num(my_old_X)    
    
    X = np.reshape(np.array(my_X),(my_X.shape[0],my_X.shape[1]))
    Y = np.reshape(np.array(y),(y.shape[0],1))
        
    from sklearn.model_selection import train_test_split  
        
    train_x, val_x, train_y,val_y = train_test_split(X, Y, test_size=0.2, random_state=0) 
    
    train_x, test_x, train_y,test_y = train_test_split(train_x, train_y, test_size=0.2, random_state=0) 
 
    train_X = np.delete(train_x, 0, axis=1)
    final_train_X = feature_scaling(train_X)
    
    val_X = np.delete(val_x, 0, axis=1)
    final_val_X = feature_scaling(val_X)
    
    test_X = np.delete(test_x, 0, axis=1)
    final_test_X = feature_scaling(test_X)
    
        
    print("Shape of train-X-data:",final_train_X.shape)
    print("Shape of train-Y-data:",train_y.shape)

    print("Shape of val-X-data:",final_val_X.shape)
    print("Shape of val-Y-data:",val_y.shape)
    
    print("Shape of test-X-data:",final_test_X.shape)
    print("Shape of test-Y-data:",test_y.shape)
    
    linear_regression_simple(final_train_X,train_y,final_val_X,val_y,final_test_X,test_y,w,b)
    
model()
