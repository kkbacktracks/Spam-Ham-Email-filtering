import sys
import MNB
import LR_MCAP_L2
import SGD
import DNB
import feature_bag_of_words
import feature_bernoulli_model




arguments = list(sys.argv)

dataset = arguments[1]
algorithm = arguments[2]

def start_algo():

    hsh = {'mnb': 'Multinomial Naive Bayes', 'dnb': 'Discrete Naive Bayes', 'lr': 'Logistic Regression', 'sgd': 'Stochastic Gradient Descent', 'bow':'Bag of Words', 'bern': 'Bernoulli'}
    
    try:
        feature_model = arguments[3]
    except:
        pass
    
    if algorithm == 'mnb':
        result = MNB.evaluate_MNB(dataset)
            
    elif algorithm == 'dnb':
        result = DNB.evaluate_DNB(dataset)
        
    elif algorithm == 'lr':
        if feature_model == 'bow':
            result = LR_MCAP_L2.evaluate_LRMCAP_bow(dataset)
        elif feature_model == 'bern':
            result = LR_MCAP_L2.evaluate_LRMCAP_bernoulli(dataset)
        else:
            print('Incorrect Argument')
            
    elif algorithm == 'sgd':
        if feature_model == 'bow':
            result = SGD.evaluate_SGD_bow(dataset)
        elif feature_model == 'bern':
            result = SGD.evaluate_SGD_bernoulli(dataset)
        else:
            print('Incorrect Argument')
            
    elif algorithm == 'matrix_bow':
        result = feature_bag_of_words.get_matrix_bow(dataset)
        print('Results for Bag Of Words Matrix on dataset :', dataset)
        print("Train - Spam Matrix", result[0])
        print("Train - Ham Matirx", result[1])
        print("Test - Spam Matrix", result[2])
        print("Test - Ham Matrix", result[3])
        
        return result
    
    elif algorithm == 'matrix_bern':
        result = feature_bernoulli_model.get_matrix_bow(dataset)
        print('Results for Bernoulli Matrix on dataset :', dataset)
        print("Train - Spam Matrix", result[0])
        print("Train - Ham Matirx", result[1])
        print("Test - Spam Matrix", result[2])
        print("Test - Ham Matrix", result[3])
        
        return result
            
    else:
        print('Incorrect Argument')
        
    
    print('Results for ', hsh[algorithm], 'algorithm on dataset :', dataset)
    print("Accuracy", result[0])
    print("Precision", result[1])
    print("Recall", result[2])
    print("F1 Score", result[3])
    
    
if __name__ == '__main__':
    result = start_algo()
    
    
    

