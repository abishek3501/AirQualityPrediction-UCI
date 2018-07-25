# AirQualityPrediction-UCI
An end-to-end complete multi-variate linear regression with polynomial features and a 3-layer Deep Neural Network regression model to predict the Air Quality using the Air Quality Dataset from UCI ML repo. 

Used Tensorflow and scikit-learn for implementing.

It is seen that, keeping all the other factors of learning common between these two models, if we use higher order polynomial (more complex features), the DNN outperforms the multivariate regression model and if the not-so-higer-order-polynomials (lesser complex features), then the multivariate polynomial regression model outperforms the 3-layer DNN. 

This shows how DNN performs better when we have a larger dataset, more complex feautres, i.e, big data. Whereas simpler regression modules performs well for small amount of data but it generally does not generalize well into the big-data era.

