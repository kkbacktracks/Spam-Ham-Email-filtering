from math import log10 as log
import feature_bernoulli_model
import evaluation_metrics


def evaluate_DNB(dataset):
    model_bernoulli_spam_emails, model_bernoulli_ham_emails, spam_dict_all_docs, ham_dict_all_docs, files_dict = feature_bernoulli_model.model_bernoulli_feature(dataset, True)
    
    conditional_prior, conditional_prob, cond_prob_word_not_in_doc = train_DNB(model_bernoulli_spam_emails, model_bernoulli_ham_emails, spam_dict_all_docs, ham_dict_all_docs, files_dict)
    
    model_bernoulli_spam_emails2, model_bernoulli_ham_emails2, spam_dict_all_docs2, ham_dict_all_docs2, files_dict2 = feature_bernoulli_model.model_bernoulli_feature(dataset, False)
    
    
    predicted_spam = []
    for doc in model_bernoulli_spam_emails2:
        predicted_spam.append(test_DNB(conditional_prior, conditional_prob, cond_prob_word_not_in_doc, doc))
    
    actual_spam = [1] * len(model_bernoulli_spam_emails2)
    
    
    predicted_ham = []
    for doc in model_bernoulli_ham_emails2:
        predicted_ham.append(test_DNB(conditional_prior, conditional_prob, cond_prob_word_not_in_doc, doc))
    
    actual_ham = [0] * len(model_bernoulli_ham_emails2)
    
    
    combine_actual = actual_ham + actual_spam
    combine_predict = predicted_ham + predicted_spam 
   
    accuracy = evaluation_metrics.accuracy(combine_actual, combine_predict)
    precision = evaluation_metrics.precision(combine_actual, combine_predict)
    recall = evaluation_metrics.recall(combine_actual, combine_predict)
    f1_score = evaluation_metrics.f1_score(recall, precision)
    
    return accuracy, precision, recall, f1_score
        
    

def train_DNB(model_bernoulli_spam_emails, model_bernoulli_ham_emails, spam_dict_all_docs, ham_dict_all_docs, files_dict):
    total_docs_count = len(model_bernoulli_spam_emails) + len(model_bernoulli_ham_emails)
    
    conditional_prior = {}
    conditional_prob = {}
    conditional_prob["spam"] = {}
    conditional_prob["ham"] = {}
    
    cond_prob_word_not_in_doc = {}
    cond_prob_word_not_in_doc["spam"] = {}
    cond_prob_word_not_in_doc["ham"] = {}
    
    len_spam = len(model_bernoulli_spam_emails)
    len_ham = len(model_bernoulli_ham_emails)
    
    conditional_prior["spam"] = log(len(model_bernoulli_spam_emails)/ float(total_docs_count))
    conditional_prior["ham"] = log(len(model_bernoulli_ham_emails)/ float(total_docs_count))
    
    for word in spam_dict_all_docs:
        conditional_prob["spam"][word] = log(1+spam_dict_all_docs[word]/(float(len_spam+2)))
        
    for word in ham_dict_all_docs:
        conditional_prob["ham"][word] = log(1+ham_dict_all_docs[word]/(float(len_ham+2)))
        
    cond_prob_word_not_in_doc["spam"] = log(1/(float(len(model_bernoulli_spam_emails)+2)))
    cond_prob_word_not_in_doc["ham"] = log(1/(float(len(model_bernoulli_ham_emails)+2)))
    
    return conditional_prior, conditional_prob, cond_prob_word_not_in_doc

    
def test_DNB(conditional_prior, conditional_prob, cond_prob_word_not_in_doc, email_words):
    final_probability = {}
    for clas in list(conditional_prior):
        final_probability[clas] = conditional_prior[clas]
        for word in list(email_words):
            if email_words[word]!=0:
                try:
                    final_probability[clas]+=conditional_prob[clas][word]
                except KeyError:
                    final_probability[clas]+=cond_prob_word_not_in_doc[clas]
                
    if final_probability["spam"]>final_probability["ham"]:
        return 1
    else:
        return 0
    
    
#predicted_spam, predicted_ham, accuracy, precision, recall, f1_score = evaluate_DNB('Datasets')
                
                                         
                                               