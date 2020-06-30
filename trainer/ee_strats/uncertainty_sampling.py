import numpy as np
import random

# Module referencing Exploration/Exploitation strategies
# The must have "list_of_preds_dicts" as input 
# Each of the dicts of the list have the keys "score" and "filename" 
# You can use more arguments and then use partial functions in the config.py file to set the arguments
# =======================================================================

def sort_decreasingly(list_of_preds_dicts, uncertainty_measure):
    '''Help function to sort the dict decreasingly according to an uncertainty measure'''
    for img_dict in list_of_preds_dicts:
        img_dict["score"] = uncertainty_measure(img_dict["score"])
    return sorted(list_of_preds_dicts, key = lambda i: (i["score"]), reverse=True)

def entropy(list_of_preds_dicts):
    '''Orders the predictions by entropy sampling'''
    def sum_entropy(pred):
        '''Calculate the sum of the entropies of each probabilites for a given prediction, returning the uncertainty score'''
        def entropy(prob):
            if prob==0:
                return 0
            return -prob*np.log(prob)
        entropy_vect = np.vectorize(entropy)
        return np.sum(entropy_vect(pred), dtype=np.float64)
    return sort_decreasingly(list_of_preds_dicts, sum_entropy)

def least_confidence(list_of_preds_dicts):
    ''' Orders the predictions by least confidence sampling '''  
    def lc(pred):
        '''Get the uncertainty score using least confidence'''
        return 1-pred.max()
    return sort_decreasingly(list_of_preds_dicts, lc)

def margin_of_confidence(list_of_preds_dicts):
    '''Orders the predictions by margin of confidence sampling'''
    def moc(pred):
        sorted_pred = np.sort(pred)
        return 1 - (sorted_pred[0] - sorted_pred[1])
    return sort_decreasingly(list_of_preds_dicts, moc)

def ratio_of_confidence(list_of_preds_dicts):
    '''Orders the predictions by ratio of confidence sampling'''
    def roc(pred):
        sorted_pred = np.sort(pred)
        if sorted_pred[0]==0:
            return 1
        else:
            return sorted_pred[1]/sorted_pred[0]
    return sort_decreasingly(list_of_preds_dicts, roc)

def randomize_sampling(list_of_preds_dicts, sampling, p):
    '''EE strategy combining an uncertainty sampling method with random sampling in order to not get stuck around the decision boundary 
        p represents the probability to exploit the data, 1-p to explore.'''
    sampled_list = sampling(list_of_preds_dicts)
    random_list = list_of_preds_dicts[:]
    random.shuffle(random_list)
    del list_of_preds_dicts
    out = []
    for sampled_dict, random_dict in zip(sampled_list, random_list):
        t = random.random()
        if t<=p:
            out.append(sampled_dict)
        else:
            out.append(random_dict)
    return out

