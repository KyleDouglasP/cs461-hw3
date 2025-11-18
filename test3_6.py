import math
import numpy as np

# Given a variable, mean, and stdev, computes the probability of the variable based on a normal distribution (reused code from HW1)
def multivariate_gaussian(variable, mean, COV) -> float:
    N = mean.shape[0]
    variable_centered = variable-mean
    norm = 1/math.sqrt((2*math.pi)**N*(np.linalg.det(COV)))
    exp = -0.5 * variable_centered.T @ np.linalg.pinv(COV) @ variable_centered
    return norm*math.exp(exp)

MEAN_POS = np.array([0, 0])
MEAN_NEG_1 = np.array([0, 2])
MEAN_NEG_2 = np.array([0, -2])
I = np.eye(2) # 2x2 Identity for distributions

# Given a 2D variable, classifies it utilizing the GDA decision rule, using the known densities from 3.6
def classifier(variable) -> int:
    
    pos_gaussian = multivariate_gaussian(variable, MEAN_POS, I)
    neg_gaussian = 0.5*multivariate_gaussian(variable, MEAN_NEG_1, I)+0.5*multivariate_gaussian(variable, MEAN_NEG_2, I)

    if pos_gaussian>=neg_gaussian: return 1
    else: return -1

test_x=np.load("P3_data/data_2/test.npz")["x"]
test_y=np.load("P3_data/data_2/test.npz")["y"]

correct = 0
count = 0

for num,x in enumerate(test_x):
    classification = classifier(x)
    if classification==test_y[num]: correct+=1
    count+=1

print(f'Testing accuracy: {100*(correct/count)}%')