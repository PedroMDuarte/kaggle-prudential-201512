# Prudential Life Insurance Assessment

Made my final submission on 2016-02-14, with a score of 0.66551.  At that time the leader is Carlos Fernandez with a score of 0.68325. 

I used the xgboost python library for my submission.  I bagged a set of three classifiers (objective='multi:softmax') with a set of 12 regressors (objective='reg:linear').   The bagged model was used to predict a score continuous variable.  The score was then binned into categories according to a set of best cutoff values.  The best cutoff values were obtained by maximizing the qwk score for each of the models in the final bag, and then averaging the optimized cutoffs for all of the models.  

This was my first Kaggle competition, learned a lot while doing it, and looking forward to applying all the experience to further work. 
