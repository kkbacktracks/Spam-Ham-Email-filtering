Step 1:

	You will need the following modules 
	```
	1 import sys
	2 import copy 
	3 import pandas
	4 import os 
	5 import re
	6 from collections import Counter 
	7 import random
	8 import numpy as np
	9 from decimal import Decimal 
	10 from math import log10 as log
	11 from sklearn.linear_model import SGDClassifier 
	12 from sklearn.model_selection import GridSearchCV 
	
	```


Step 2:

	Please enter the following command line argument:-
	
	python hw1_main.py [dataset_name] [algorithm_name] [type_of_model]

	Please use the following command line parameters for the main.py file :-
	* ***Dataset name***
		Provide the name of folder for the dataset
		Dataset folder names : hw1, enron1, enron4
	* ***Algorithm name*** 
		* matrix_bow - To get the matrix of Bag of words representation (3rd argument type_of_model is not required for this algorithm) 
		* matrix_bern - To get the matrix of Bernoulli representation (3rd argument type_of_model is not required for this algorithm) 
		* mnb - for the multi-nomial naive Bayes (3rd argument type_of_model is not required for this algorithm) 
		* dnb - for the discrete naive Bayes (3rd argument type_of_model is not required for this algorithm) 
		* lr - for the Logistic Regression (MCAP) (3rd argument required) 
		* sgd - for the SGD classifier (3rd argument required)
		
	* ***Type of model***
		This is used only for the -lr and -sgd algorithms(2nd parameter) 
		* bow - use this parameter for choosing the bag of words model 
		* bern - use this for the Bernoulli model
		
	Example Commands : 
	1) python hw1_main.py enron1 mnb
	2) python hw1_main.py enron4 lr bern
	3) python hw1_main.py hw1 sgd bow
	4) python hw1_main.py enron1 matrix_bow

Step 3:
	Find the complete report of this project with the name Final_Report.pdf
