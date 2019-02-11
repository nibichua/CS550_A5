'''
Created on Apr 15, 2018

@author: mroch

    Name: Tony La and Shawn Nehemiah Chua
    RedID: 817862169 and 817662151
    Class Information: CS550: Artificial Intelligence, Spring 2018
    Professor: Professor Marie Roch
    Assignment 5: Machine Learning
    Due Date: 5/1/2018
    Filename: driver.py
'''
from ml_lib.learning import (DataSet, 
                             DecisionTreeLearner, NeuralNetLearner)
from std_cv import cross_validation
from random import shuffle

from copy import deepcopy
    
def learn(dataset):
    shuffle_data(dataset)
    #perform cross validation using decision tree and neural net
    csv_dtl = cross_validation(DecisionTreeLearner,dataset)
    dataset.attributes_to_numbers()
    
    ''' for formatting purposes'''
    learners = ()
    if dataset.name is not 'restaurant':
        learners = ('\t\tDecisionTreeLearner','\t\tNeuralNetLearner')
    else:
        learners = ('\tDecisionTreeLearner','\tNeuralNetLearner')
    ''' for output formatting purposes'''
        
    list = " ".join([str(x) for x in csv_dtl[2]])
    print(csv_dtl[0]+'\t   '+csv_dtl[1]+'  '+list+'\t'+csv_dtl[3]+learners[0])
    
    if dataset.name is not 'abalone':
        csv_nnl = cross_validation(NeuralNetLearner,dataset)
        list = " ".join([str(x) for x in csv_nnl[2]])
        print(csv_nnl[0]+'\t   '+csv_nnl[1]+'  '+list+'\t'+csv_nnl[3]+learners[1])    

def shuffle_data(dataset):
    shuffle(dataset.examples)   
    
def main():
    print('Mean\t   StdDev Errors for each fold\t\t\t\t\t\tCorpus\t\tLearner')
    for dataset in ['iris','orings','restaurant','zoo','abalone']:
        data = DataSet(name=dataset)
        learn(data)
    
if __name__ == '__main__':
    main()
