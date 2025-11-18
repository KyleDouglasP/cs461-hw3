import math
import numpy as np

# Given a variable, mean, and stdev, computes the probability of the variable based on a normal distribution (reused code from HW1)
def gaussian(variable: float, mean: float, variance: float) -> float:
    return (1/math.sqrt(2*math.pi*variance))*math.exp(-(((variable-mean)**2)/(2*variance)))

MEAN_POS = -0.0721922106722285
VAR_POS = 1.3096715040939155
MEAN_NEG = 0.9401561132214228
VAR_NEG = 1.9437063405522659

# Given a variable, classifies it utilizing the decision rule given in 3.1
def classifier(variable: float) -> int:
    
    pos_gaussian = gaussian(variable, MEAN_POS, VAR_POS)
    neg_gaussian = gaussian(variable, MEAN_NEG, VAR_NEG)

    if pos_gaussian>=neg_gaussian: return 1
    else: return -1

test_x=np.load("P3_data/data_1/test.npz")["x"]
test_y=np.load("P3_data/data_1/test.npz")["y"]

correct = 0
count = 0

for num,x in enumerate(test_x):
    classification = classifier(x)
    if classification==test_y[num]: correct+=1
    count+=1

print(f'Testing accuracy: {100*(correct/count)}%')