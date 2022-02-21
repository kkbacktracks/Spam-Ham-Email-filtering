import os
import re
import nltk
from nltk.corpus import stopwords
import copy
import pandas as pd

def import_train_or_test_data(dataset, isTrain):
    dirname = os.path.dirname(__file__)
    
    ham_files = []
    spam_files = []
    all_files = ""

    
    path_of_text_files = os.path.join(dirname, dataset)
    
    if isTrain:
        path_of_text_files = os.path.join(path_of_text_files, "train")
    else:
        path_of_text_files = os.path.join(path_of_text_files, "test")
        
    path_of_ham_files = os.path.join(path_of_text_files, "ham")
    path_of_spam_files = os.path.join(path_of_text_files, "spam")
    
    read_files(spam_files, path_of_spam_files)
    read_files(ham_files, path_of_ham_files)
    
    all_files = " ".join(spam_files)
    all_files = all_files + " ".join(ham_files)
    
    
    return spam_files,ham_files, all_files
    
    
def read_files(list_values, path):
    os.chdir(path)
    for file in os.listdir():
       
        if file.endswith(".txt"):
            file_path = f"{path}\{file}"
      
            read_text_file(file_path, list_values)
    
       
def read_text_file(file_path, list_values):
    try:
        with open(file_path, 'r', encoding="utf-8") as f:
            list_values.append(f.read())
           
    except:
       pass

def get_matrix_bow(dataset):
    spam_files_train, ham_files_train, all_files_train = import_train_or_test_data(dataset, True)
    spam_files_test, ham_files_test, all_files_test = import_train_or_test_data(dataset, False)
    
    df_train_spam, df_train_ham = get_final_matrix(spam_files_train, ham_files_train, all_files_train)
    df_test_spam, df_test_ham = get_final_matrix(spam_files_test, ham_files_test, all_files_test)
    
    return df_train_spam, df_train_ham, df_test_spam, df_test_ham

def get_final_matrix(spam_files, ham_files, all_files):
    files_dict = {}
    all_words_in_files = re.findall("[a-zA-Z]+", all_files)
    #nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))
    
    for word in all_words_in_files:
        word = word.lower()
        
        if word in files_dict:
            continue
        else:
            if word not in stop_words:
                files_dict[word] = 0
        
    
    model_bernoulli_spam_emails = []
    spam_dict_all_docs = {}
    
    for email in spam_files:
        dictionary_temp = copy.deepcopy(files_dict)
        email1 = re.findall("[a-zA-Z]+", email)
        
        for each_word in email1:
            each_word = each_word.lower()
            
            if each_word in dictionary_temp:
                dictionary_temp[each_word] = 1
                spam_dict_all_docs[each_word] = 1
        
        model_bernoulli_spam_emails.append(dictionary_temp)
        
    df1 = pd.DataFrame(model_bernoulli_spam_emails)
    df1['emails'] = spam_files
    first_column = df1.pop('emails')
    df1.insert(0, 'emails', first_column)
        
    model_bernoulli_ham_emails = []
    ham_dict_all_docs = {} 
            
    for email in ham_files:
        email1 = re.findall("[a-zA-Z]+", email)
        dictionary_temp = copy.deepcopy(files_dict)
        
        for each_word in email1:
            each_word = each_word.lower()
            
            if each_word in dictionary_temp:
                dictionary_temp[each_word] = 1
                ham_dict_all_docs[each_word] = 1
        
        model_bernoulli_ham_emails.append(dictionary_temp)
        
    df2 = pd.DataFrame(model_bernoulli_ham_emails)
    df2['emails'] = ham_files
    first_column = df2.pop('emails')
    df2.insert(0, 'emails', first_column)
    
    return df1, df2
    
        
def model_bernoulli_feature(dataset, isTrain):
    spam_files, ham_files, all_files = import_train_or_test_data(dataset, isTrain)
    
    files_dict = {}
    all_words_in_files = re.findall("[a-zA-Z]+", all_files)
    #nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))
    
    for word in all_words_in_files:
        word = word.lower()
        
        if word in files_dict:
            continue
        else:
            if word not in stop_words:
                files_dict[word] = 0
        
    
    model_bernoulli_spam_emails = []
    spam_dict_all_docs = {}
    
    for email in spam_files:
        dictionary_temp = copy.deepcopy(files_dict)
        email1 = re.findall("[a-zA-Z]+", email)
        
        for each_word in email1:
            each_word = each_word.lower()
            
            if each_word in dictionary_temp:
                dictionary_temp[each_word] = 1
                spam_dict_all_docs[each_word] = 1
        
        model_bernoulli_spam_emails.append(dictionary_temp)
        
    model_bernoulli_ham_emails = []
    ham_dict_all_docs = {} 
            
    for email in ham_files:
        email1 = re.findall("[a-zA-Z]+", email)
        dictionary_temp = copy.deepcopy(files_dict)
        
        for each_word in email1:
            each_word = each_word.lower()
            
            if each_word in dictionary_temp:
                dictionary_temp[each_word] = 1
                ham_dict_all_docs[each_word] = 1
        
        model_bernoulli_ham_emails.append(dictionary_temp)    
        
    return model_bernoulli_spam_emails, model_bernoulli_ham_emails, spam_dict_all_docs, ham_dict_all_docs, files_dict
    


