import feature_bernoulli_model
import feature_bag_of_words
import evaluation_metrics
import random
import numpy as np

from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV


def evaluate_SGD_bow(dataset):
    model_bernoulli_spam_emails, model_bernoulli_ham_emails, count_text_in_all_docs,spam_dict_all_docs, ham_dict_all_docs, files_dict = feature_bag_of_words.model_bow_feature(dataset, True)
    
    model_bernoulli_spam_emails2, model_bernoulli_ham_emails2, count_text_in_all_docs, spam_dict_all_docs2, ham_dict_all_docs2, files_dict2 = feature_bag_of_words.model_bow_feature(dataset, False)
    
    
    
    training_data, validation_data = split_data(model_bernoulli_spam_emails, model_bernoulli_ham_emails, True)
    
    testing_data = split_data(model_bernoulli_spam_emails, model_bernoulli_ham_emails, False)
    
    features = list(training_data[0])
    
    train_x_arr, train_y_arr = convert_data_for_SGD(training_data, features)
    test_x_arr, test_y_arr = convert_data_for_SGD(testing_data, features)
    validation_x_arr, validation_y_arr = convert_data_for_SGD(validation_data, features)
    
    classifier = tune_params(validation_x_arr, validation_y_arr)
    
    #training the model
    trained_model = classifier.fit(train_x_arr, train_y_arr)
    
    #testing the model
    predicted_class = test_SGD(trained_model, test_x_arr)
    
        
    accuracy = evaluation_metrics.accuracy(test_y_arr, predicted_class)
    precision = evaluation_metrics.precision(test_y_arr, predicted_class)
    recall = evaluation_metrics.recall(test_y_arr, predicted_class)
    f1_score = evaluation_metrics.f1_score(recall, precision)
    
    return accuracy, precision, recall, f1_score, predicted_class
        
def evaluate_SGD_bernoulli(dataset):
        model_bernoulli_spam_emails, model_bernoulli_ham_emails,spam_dict_all_docs, ham_dict_all_docs, files_dict = feature_bernoulli_model.model_bernoulli_feature(dataset, True)
        
        model_bernoulli_spam_emails2, model_bernoulli_ham_emails2, spam_dict_all_docs2, ham_dict_all_docs2, files_dict2 = feature_bernoulli_model.model_bernoulli_feature(dataset, False)
        
        
        
        training_data, validation_data = split_data(model_bernoulli_spam_emails, model_bernoulli_ham_emails, True)
        
        testing_data = split_data(model_bernoulli_spam_emails, model_bernoulli_ham_emails, False)
        
        features = list(training_data[0])
        
        train_x_arr, train_y_arr = convert_data_for_SGD(training_data, features)
        test_x_arr, test_y_arr = convert_data_for_SGD(testing_data, features)
        validation_x_arr, validation_y_arr = convert_data_for_SGD(validation_data, features)
        
        classifier = tune_params(validation_x_arr, validation_y_arr)
        
        #training the model
        trained_model = classifier.fit(train_x_arr, train_y_arr)
        
        #testing the model
        predicted_class = test_SGD(trained_model, test_x_arr)
        
            
        accuracy = evaluation_metrics.accuracy(test_y_arr, predicted_class)
        precision = evaluation_metrics.precision(test_y_arr, predicted_class)
        recall = evaluation_metrics.recall(test_y_arr, predicted_class)
        f1_score = evaluation_metrics.f1_score(recall, precision)
        
        return accuracy, precision, recall, f1_score



def split_data(spam_data, ham_data, isTrain):
    for dict in spam_data:
        dict["class_spam_ham"] = 1
    for dict in ham_data:
        dict["class_spam_ham"] = 0
        
        
    combine_email = spam_data + ham_data
    if not isTrain:
        return combine_email
    else:
        random.shuffle(combine_email)
        
        train, validate = combine_email[0:int(len(combine_email) * 0.70)], combine_email[int(len(combine_email)*0.70):-1]
        
        return train, validate
    
def convert_data_for_SGD(data, features):
    x_arr, y_arr = [], []
    
    for each in data:
        x_current_doc = []
        y_arr.append(each['class_spam_ham'])
        
        for word in features:
            try:
                x_current_doc.append(each[word])
            except:
                x_current_doc.append(0)
                
        x_arr.append(x_current_doc)
        
    return x_arr, y_arr

def tune_params(validation_x_arr, validation_y_arr):
    
    params = {'alpha' : (0.01, 0.05),
              'max_iter' : (range(500, 3000, 1000)),
              'learning_rate': ('optimal', 'invscaling', 'adaptive'),
              'eta0' : (0.3, 0.7),
              'tol' : (0.001, 0.005)}
    
    sgd_classifier = SGDClassifier()
    
    gridSearch = GridSearchCV(sgd_classifier, params, cv=5)
    gridSearch.fit(validation_x_arr, validation_y_arr)
    
    return gridSearch

def test_SGD(classifier, test_x_arr):
    predicted_class = []
    
    for each in test_x_arr:
        predicted_class.append(classifier.predict(np.reshape(each, (1,-1))))
        
    return predicted_class


#accuracy, precision, recall, f1_score, predicted_class = evaluate_SGD_bow('enron4')
    