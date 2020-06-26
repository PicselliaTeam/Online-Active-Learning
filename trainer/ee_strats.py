import numpy as np
import random

# Module referencing Exploration/Exploitation strategies
# The must have "list_of_preds_dicts" as input 
# Each of the dicts of the list have the keys "score" and "filename" 
# You can use more arguments and then use partial functions in the config.py file to set the arguments
# =======================================================================

def entropy_sampling(list_of_preds_dicts):
    '''Exploration only strategy based on the entropy of a prediction'''
    def sum_entropy(pred):
    '''Calculate the sum of the entropies of each probabilites for a given prediction'''
        def entropy(prob):
            if prob==0:
                return 0
            return -prob*np.log(prob)
        entropy_vect = np.vectorize(entropy)
        return np.sum(entropy_vect(pred), dtype=np.float64)

    for img_dict in list_of_preds_dicts:
        img_dict["score"] = sum_entropy(img_dict["score"])
    return sorted(list_of_preds_dicts, key = lambda i: (i["score"]), reverse=True)


def random_entropy_sampling(list_of_preds_dicts, p)
    '''EE strategy combining Entropy Sampling (exploration) with random sampling (exploitation)
        p represents the probability to explore the data, 1-p to exploit'''
    entropy_list = entropy_sampling(list_of_preds_dicts)
    random_list = list_of_preds_dicts[:]
    random.shuffle(random_list)
    del list_of_preds_dicts
    out = []
    for entropy_dict, random_dict in zip(entropy_list, random_list):
        t = random.random()
        if t<=p:
            out.append(entropy_dict)
        else:
            out.append(random_dict)
    return out

