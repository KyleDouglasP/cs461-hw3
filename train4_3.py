import numpy as np

STEP_SIZE = 1e-4
NLL_LIMIT = 10

def sigmoid(z):
    return 1/(1+np.exp(-z))

def negative_log_likelihood(data, labels, weights):
    predict = sigmoid(data @ weights)
    return np.sum(-labels*np.log(predict)-(1-labels)*np.log(1-predict))

def gradient(data, labels, weights):
    predict = sigmoid(data @ weights)
    return data.T @ (predict - labels)    

train_data=np.load("P4_files/train4_2.npz")["data"]
train_labels=np.load("P4_files/train4_2.npz")["labels"]

N, D = train_data.shape
weights = np.zeros(D)
prev_nll = negative_log_likelihood(train_data, train_labels, weights)

for i in range(100000):
    grad = gradient(train_data, train_labels, weights)
    weights -= STEP_SIZE*grad
    nll = negative_log_likelihood(train_data, train_labels, weights)

    if abs(prev_nll - nll) < NLL_LIMIT:
        break

train_correct = 0
for num,x in enumerate(train_data):
    prob = sigmoid(x @ weights)
    if prob>=.5: prob=1
    else: prob=0
    if prob==train_labels[num]: train_correct+=1

print(f'Training accuracy: {100*(train_correct/train_labels.size)}%')

test_data=np.load("P4_files/test4_2.npz")["data"]
test_labels=np.load("P4_files/test4_2.npz")["labels"]

test_correct = 0
for num,x in enumerate(test_data):
    prob = sigmoid(x @ weights)
    if prob>=.5: prob=1
    else: prob=0
    if prob==test_labels[num]: test_correct+=1

print(f'Test accuracy: {100*(test_correct/test_labels.size)}%')

mail_test=np.load("P4_files/mail50D.npz")["data"]
mail_test_prob = sigmoid(mail_test @ weights)
if mail_test_prob>=.5: print("mail.txt is spam.")
else: print("mail.txt is not spam.")