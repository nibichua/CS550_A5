'''
    Name: Tony La and Shawn Nehemiah Chua
    RedID: 817862169 and 817662151
    Class Information: CS550: Artificial Intelligence, Spring 2018
    Professor: Professor Marie Roch
    Assignment 5: Machine Learning
    Due Date: 5/1/2018
    Filename: std_cv.py
'''
from statistics import (stdev, mean)
from ml_lib.learning import (err_ratio, train_and_test)


def cross_validation(learner, dataset, k=10):
    """Perform k-fold cross_validation
    Run k trials where each trial has a different (k-1)/k percentage
    of the data as training data and 1/k as test data.
    
    Returns tuple (mean_err, std_err, fold_errors, models)
    """
    if k is None:
        k = len(dataset.examples)

    fold_errT = 0   # fold error on training data
    fold_errV = 0   # fold error on validation data
    
    n = len(dataset.examples)
    examples = dataset.examples
    
    #initialize as empty list for both
    #error_fold_numbers is for dealing with numberical values such as ints and floats
    #error_fold_rates are numerical values used for functions below
    error_fold_rates = []
    error_fold_numbers = []
    
    for fold in range(k):   # for each fold
        # Split into train and test
        # Note that this is not a canonical cross validation where
        # every pieces of data is used for training and testing
        # due to the shuffling above.
        train_data, val_data = train_and_test(dataset, fold * (n / k),
                                              (fold + 1) * (n / k))
        dataset.examples = train_data
        h = learner(dataset)
        # predict and accumulate the error rate on 
        # the training and validation data
        fold_errT += err_ratio(h,dataset,train_data)
        
        fold_err_validation = err_ratio(h,dataset,val_data)
        fold_errV += err_ratio(h, dataset, val_data)
        
        '''rounding to three decimal places'''
        error_fold_rates.append(format(fold_err_validation,'.3f'))
        error_fold_numbers.append(fold_err_validation)
        # Reverting back to original once test is completed
        dataset.examples = examples
        
    # Return average per fold rates
    mean_error_rate = round(mean(error_fold_numbers),3)
    #return standard deviation
    stdev_error_rate = round(stdev(error_fold_numbers,mean_error_rate),3)
    
    return format(mean_error_rate,'.3f'),format(stdev_error_rate,'.3f'),error_fold_rates, dataset.name