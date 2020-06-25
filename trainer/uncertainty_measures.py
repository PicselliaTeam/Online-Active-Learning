import numpy as np



def SumEntropy(pred):
    '''Calculate the sum of the entropies of each probabilites for a prediction'''
    def Entropy(prob):
        if prob==0:
            return 0
        return -prob*np.log(prob)
    Entropy_vect = np.vectorize(Entropy)
    return np.sum(Entropy_vect(pred), dtype=np.float64)
