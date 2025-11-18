import numpy as np

# Taking the entire csv as a (4000,2002) matrix
data = np.genfromtxt("P4_files/spam_ham.csv", delimiter=",", skip_header=1)

labels = data[:, -1] # Take the labels as the last column of the data
data = data[:, 1:-1] # Get rid of the first column (row number) and last column (labels)

mean = np.sum(data, axis=0, keepdims=True) / data.shape[0]

data_centered = data-mean
cov = np.matmul(data_centered.T, data_centered) / (data_centered.shape[0] - 1)

# Eigen decomposition of the covariance matrix
eigenvalues, eigenvectors = np.linalg.eigh(cov)

# Sort eigenvalues and eigenvectors for PCA
sorted_indices = np.argsort(eigenvalues)[::-1]  # Sort in descending order
eigenvalues = eigenvalues[sorted_indices]
eigenvectors = eigenvectors[:, sorted_indices]

fifty_dimensional_data = np.dot(data_centered, eigenvectors[:, :50]) * (1 / np.sqrt(eigenvalues[:50]))

train_data = fifty_dimensional_data[:3500]
train_labels = labels[:3500]
test_data = fifty_dimensional_data[3500:]
test_labels = labels[3500:]

np.savez("P4_files/train4_2.npz", data=train_data, labels=train_labels)
np.savez("P4_files/test4_2.npz", data=test_data, labels=test_labels)

mail_data = np.load("P4_files/mail.npz")["data"]
fifty_dimensional_mail_data = np.dot(mail_data-mean, eigenvectors[:, :50]) * (1 / np.sqrt(eigenvalues[:50]))
np.savez("P4_files/mail50D.npz", data=fifty_dimensional_mail_data)