import numpy as np

train_x=np.load("P3_data/data_1/train.npz")["x"]
train_y=np.load("P3_data/data_1/train.npz")["y"]

positive_x = []
negative_x = []

for num,y in enumerate(train_y):
    if y==1: positive_x.append(train_x[num])
    elif y==-1: negative_x.append(train_x[num])

print(f'P[+] = {len(positive_x)/(len(positive_x)+len(negative_x))}')
print(f'Positive Sample Mean: {np.mean(positive_x)}')
print(f'Positive Sample Variance: {np.var(positive_x, ddof=1)}')

print(f'P[-] = {len(negative_x)/(len(positive_x)+len(negative_x))}')
print(f'Negative Sample Mean: {np.mean(negative_x)}')
print(f'Negative Sample Variance: {np.var(negative_x, ddof=1)}')