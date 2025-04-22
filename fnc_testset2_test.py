import numpy as np
from sklearn.metrics import balanced_accuracy_score

def loadPredictions(dataFolder):
    idTest = [1,2,3,4]
    
    yPred = []
    for k,id in enumerate(idTest):
        yv = np.genfromtxt('{}trial{:02d}.y.v.csv'.format(dataFolder,id),delimiter=',')
        yPred.append(yv)

    return yPred

dataFolder = 'Test2-Pred/'
len_test = [8577, 8619, 12035, 9498]

def test_length():
    yPred = loadPredictions(dataFolder)
    for i in range(4):
        assert len(yPred[i])==len_test[i], 'The length of the prediction is not correct.'
    
def test_label():
    yPred = loadPredictions(dataFolder)

    for k in range(len(yPred)):
        y_len = len(yPred[k])

        # Extracting the proportions of your predicitions
        n0 = np.sum(yPred[k]==0)
        n1 = np.sum(yPred[k]==1)
        n2 = np.sum(yPred[k]==2)
        n3 = np.sum(yPred[k]==3)
        print('Trial {:02d}: n0={:4.2f} n1={:4.2f} n2={:4.2f} n3={:4.2f}'.format(k+1,
                                            n0/y_len,n1/y_len,n2/y_len,n3/y_len))

        # Checking that things add up to 1
        assert (n0+n1+n2+n3)==y_len, 'There is something wrong with your labels.'


if __name__ == "__main__":
    test_length()
    test_label()
