import feature_bernoulli_model
import feature_bag_of_words
import evaluation_metrics
import random
import numpy as np
import copy


def evaluate_LRMCAP_bow(dataset):
    model_bernoulli_spam_emails, model_bernoulli_ham_emails, count_text_in_all_docs,spam_dict_all_docs, ham_dict_all_docs, files_dict = feature_bag_of_words.model_bow_feature(dataset, True)
    
    training_data, validation_data = data_spilt(model_bernoulli_spam_emails, model_bernoulli_ham_emails)
    
    lambda_value_trained = trainLambdaParameter( training_data, validation_data, files_dict)   
    
    training_data = training_data + validation_data
    model_weights = train_LRMCAP(training_data, lambda_value_trained, 200, files_dict, 0.01)
    
    model_bernoulli_spam_emails2, model_bernoulli_ham_emails2, count_text_in_all_docs, spam_dict_all_docs2, ham_dict_all_docs2, files_dict2 = feature_bag_of_words.model_bow_feature(dataset, False)
    
    
    predicted_spam = []
    for doc in model_bernoulli_spam_emails2:
        predicted_spam.append(test_LRMCAP(doc, model_weights))
    
    actual_spam = [1] * len(model_bernoulli_spam_emails2)
    
    
    predicted_ham = []
    for doc in model_bernoulli_ham_emails2:
        predicted_ham.append(test_LRMCAP(doc, model_weights))
    
    actual_ham = [0] * len(model_bernoulli_ham_emails2)
    
    
    combine_actual = actual_ham + actual_spam
    combine_predict = predicted_ham + predicted_spam 
   
    accuracy = evaluation_metrics.accuracy(combine_actual, combine_predict)
    precision = evaluation_metrics.precision(combine_actual, combine_predict)
    recall = evaluation_metrics.recall(combine_actual, combine_predict)
    f1_score = evaluation_metrics.f1_score(recall, precision)
    
    return accuracy, precision, recall, f1_score
        
def evaluate_LRMCAP_bernoulli(dataset):
    model_bernoulli_spam_emails, model_bernoulli_ham_emails,spam_dict_all_docs, ham_dict_all_docs, files_dict = feature_bernoulli_model.model_bernoulli_feature(dataset, True)
    
    training_data, validation_data = data_spilt(model_bernoulli_spam_emails, model_bernoulli_ham_emails)
    
    lambda_value_trained = trainLambdaParameter( training_data, validation_data, files_dict)   
    
    training_data = training_data + validation_data
    model_weights = train_LRMCAP(training_data, lambda_value_trained, 200, files_dict, 0.01)
    
    model_bernoulli_spam_emails2, model_bernoulli_ham_emails2, spam_dict_all_docs2, ham_dict_all_docs2, files_dict2 = feature_bernoulli_model.model_bernoulli_feature(dataset, False)
    
    
    predicted_spam = []
    for doc in model_bernoulli_spam_emails2:
        predicted_spam.append(test_LRMCAP(doc, model_weights))
    
    actual_spam = [1] * len(model_bernoulli_spam_emails2)
    
    
    predicted_ham = []
    for doc in model_bernoulli_ham_emails2:
        predicted_ham.append(test_LRMCAP(doc, model_weights))
    
    actual_ham = [0] * len(model_bernoulli_ham_emails2)
    
    
    combine_actual = actual_ham + actual_spam
    combine_predict = predicted_ham + predicted_spam 
   
    accuracy = evaluation_metrics.accuracy(combine_actual, combine_predict)
    precision = evaluation_metrics.precision(combine_actual, combine_predict)
    recall = evaluation_metrics.recall(combine_actual, combine_predict)
    f1_score = evaluation_metrics.f1_score(recall, precision)
    
    return accuracy, precision, recall, f1_score    



def data_spilt(spam_data, ham_data):
    for dict in spam_data:
        dict["class_spam_ham"] = 1
        dict["w0"] = 1
    for dict in ham_data:
        dict["class_spam_ham"] = 0
        dict["w0"] = 1
        
    combine_email = spam_data + ham_data
    random.shuffle(combine_email)
    
    train, validate = combine_email[0:int(len(combine_email) * 0.70)], combine_email[int(len(combine_email)*0.70):-1]
    
    return train, validate

def get_posterior_value(model_weights, data):
    result = model_weights['w0'] * 1
    
    for each_data in data:
        if each_data=='class_spam_ham' or each_data=='w0':
            continue
        else:
            if each_data in model_weights and each_data in data:
                result+=(model_weights[each_data]*data[each_data])
            
    return 1/(float(1+np.exp(-result)))
     

def train_LRMCAP(training_data, lambda_param, iter_, files_dict, learning_rate):
    model_weights = copy.deepcopy(files_dict)
    
    for weight in model_weights:
        model_weights[weight] = 0
        
    model_weights['w0'] = 0
    
    for each in range(iter_):
        for data in training_data:
            posterior_value = get_posterior_value(model_weights, data)
            
            param2_sum = 0
            
            for each_wt in model_weights:
                if data[each_wt]!=0:
                    if each_wt == 'w0':
                       param2_sum+= learning_rate*(data['class_spam_ham'] - posterior_value)
                       
                    else:
                       param2_sum+= learning_rate*(data[each_wt] * (data['class_spam_ham'] - posterior_value))
                   
                    model_weights[each_wt]+=param2_sum-learning_rate*lambda_param*model_weights[each_wt]
       
    return model_weights

def test_LRMCAP(document, model_weights):
    result = model_weights['w0'] * 1
    
    for val in document:
        if val =='w0' or val =='class_spam_ham':
            continue
        else:
            if val in model_weights and val in document:
                result+=(model_weights[val] * document[val])
            
    if result < 0:
        return 0
    else:
        return 1
    
                
        
    
    
def trainLambdaParameter(training_data, validation_data, files_dict):
    learning_rate = 0.01
    accuracy = 0.0
    final_lambda_parameter = 2
    len_validation = len(validation_data)
    
    for lambda_ in range(1, 10, 2):
        model_weights = train_LRMCAP(training_data, lambda_, 25 ,files_dict, learning_rate)
        
        class_variable = 0
        for document in validation_data:
            op = test_LRMCAP(document, model_weights)
            
            if op == document['class_spam_ham']:
                class_variable+=1
                
        temp_accuracy = class_variable / float(len_validation)
        if temp_accuracy > accuracy:
            accuracy = temp_accuracy
            final_lambda_parameter = lambda_
                           
    return final_lambda_parameter
                                         
#predicted_spam, predicted_ham, accuracy, precision, recall, f1_score = evaluate_LRMCAP_bernoulli('enron1')                                              