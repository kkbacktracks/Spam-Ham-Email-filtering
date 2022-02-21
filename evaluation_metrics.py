def accuracy(true_y, predicted_y):
    
    accuracy_count = 0
    for each in range(len(true_y)):
        if true_y[each] == predicted_y[each]:
            accuracy_count+= 1
    return accuracy_count / float(len(true_y))


def precision(true_y, predicted_y):
    
    true_positives = 0
    false_positives = 0
    for each in range(len(true_y)):
        if true_y[each] == predicted_y[each] and predicted_y[each] == 1:
            true_positives += 1
        if true_y[each] != predicted_y[each] and predicted_y[each] == 1:
            false_positives += 1
    return true_positives / float(true_positives + false_positives)


def recall(true_y, predicted_y):
    
    true_positives = 0
    false_negetives = 0
    for each in range(len(true_y)):
        if true_y[each] == predicted_y[each] and predicted_y[each] == 1:
            true_positives += 1
        if true_y[each] != predicted_y[each] and predicted_y[each] == 0:
            false_negetives += 1
    return true_positives / float(true_positives + false_negetives)


def f1_score(recall, precision):
    
    return (2 * recall * precision) / float(recall + precision)
