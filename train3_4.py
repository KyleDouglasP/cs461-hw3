import numpy as np

train_x=np.load("P3_data/data_2/train.npz")["x"]
train_y=np.load("P3_data/data_2/train.npz")["y"]

positive_x = []
negative_x = []

for num,y in enumerate(train_y):
    if y==1: positive_x.append(train_x[num])
    elif y==-1: negative_x.append(train_x[num])

positive_x = np.array(positive_x)
positive_mean = np.sum(positive_x, axis=0, keepdims=True) / positive_x.shape[0]
positive_centered = positive_x-positive_mean
positive_cov = np.matmul(positive_centered.T, positive_centered) / (positive_centered.shape[0] - 1)

print(f'P[+] = {len(positive_x)/(len(positive_x)+len(negative_x))}')
print(f'Positive Mean Vector: {positive_mean}')
print(f'Positive Covariance Matrix:\n{positive_cov}')

negative_x = np.array(negative_x)
negative_mean = np.sum(negative_x, axis=0, keepdims=True) / positive_x.shape[0]
positive_centered = negative_x-negative_mean
negative_cov = np.matmul(positive_centered.T, positive_centered) / (positive_centered.shape[0] - 1)

print(f'P[-] = {len(negative_x)/(len(positive_x)+len(negative_x))}')
print(f'Negative Mean Vector: {negative_mean}')
print(f'Negative Covariance Matrix:\n{negative_cov}')