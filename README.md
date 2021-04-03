# Credit-Card-Fraud-Analytics
Credit Card Fraud Analytics using Machine Learning Models
Fraud is an uncommon, well-considered, imperceptibly concealed, time-evolving and often carefully organized crime which appears in many types of forms.
Examples:
> Credit card Fraud
> Tax evasion
> Insurance Fraud
> Corruption
> Money Laundering
> Healthcare Fraud

In the present day scenario, fraudulent activities associated with financial transactions, particularly while using credit card, are observed to be occurring in a fast rate. One of the recent researches estimates that governments around the world lose approximately US$500 to US$600 billion annually. Hence, a fraud detection system involving various detection techniques is very much essential for the financial institutions to sustain the goodwill from the customers. Several fraud detection techniques have been proposed by researchers with application of various neural networks algorithms to find the pattern of fraud. In this project, different simple classification models are combined to form an ensemble model to improve the performance.This project is a practical approach to classify credit card transactions as normal and fraudulent.

Ensemble classification method is a supervised learning technique that combines a number of weak learners iteratively to form an efficient learner that classifies the given training samples in a more accurate way .The basic principle of ensemble method is, a set of weak learners are iteratively added to build a strong learner. Every single decision tree is a weak learner, but when they ensemble, it becomes strong learner. Each tree votes for a class and the class having maximum votes gains the final predicted class
 The classifiers predictions are combined to a meta-classifier to provide an effective performance with the help of majority voting technique. It combines the nominal outputs to predict the class label for a set of possible class label. In this technique each weak learner votes to a specific class label and the class label, which receives more than half of the votes, is the final class label. In the binary classification, the majority of the votes fetched by the classification model is considered as the better predictive model. With the help of majority voting, the hybridized classification model is developed by applying ensemble technique

> Ensemble Model’s performance can be increased by selecting diverse base estimators

The project is proposed to implement different machine learning algorithms on an imbalanced dataset such as Random forest, Logistic Regression, Naïve Bayes, Support vector machine, and Bayesian network with parallel ensemble classifier using bagging technique(Majority Voting)

CHALLENGES
1.Non- availability of real dataset
2.Size of the dataset
3.Skewness of data
4.Operational Efficiency
5.Determining appropriate evaluation parameters

DATASET

The dataset used for this project is from Kaggle which contains 284807 rows of data and 31 columns. 
Out of all the columns, the only ones that made the most sense were Time, Amount, and Class (fraud or not fraud). 
The other 28 columns were transformed using what seems to be a PCA dimensionality reduction in order to protect user identities.
Transactions were made by European cardholders in the year 2013(two days)

OBJECTIVES
1. The model should not term any genuine transaction as fraudulent(Lower False Alarm Rate)
2. FP should be minimized (Normal transactions should not be predicted as fraudulent)

ALGORITHM

Step 1 :Import the dataset
Step 2 Convert the data into data frames format
Step3: Do random sampling(under sampling)
Step 4: Decide the amount of data for training data and testing data
Step 5: Give 70% data for training and remaining data for testing(30%)
Step 6: Assign train dataset to the models
Step 7:Apply the algorithm among 6 different algorithms and create the model
Step 8:Make predictions for test dataset for each algorithm
Step9:Calculate recall and precision of each algorithm by using confusion matrix

<img width="249" alt="Before_resampling" src="https://user-images.githubusercontent.com/71324337/113470369-a7f58480-9472-11eb-92eb-c0f751482117.png">
The dataset is too imbalanced so we need to perform Random Under Sampling (deletes examples from the majority class which is the normal transactions).
<img width="555" alt="resampled_data" src="https://user-images.githubusercontent.com/71324337/113470313-4af9ce80-9472-11eb-8cfb-eea2320b023a.png">



