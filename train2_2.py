import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

import MySVM

def plot(X,Y, C=1):

    scaler = StandardScaler()
    train_x = scaler.fit_transform(X)

    model = MySVM.SVM(train_x, Y, C, kernel='linear', max_iter=600, tol=1e-5, eps=1e-5)
    # model = MySVM.SVM(train_x, Y, C=1, kernel='rbf', max_iter=600, tol=1e-5, eps=1e-5)
    model.fit()

    train_y = model.predict(train_x)

    print('support vector: {} / {}'\
        .format(len(model.alphas[model.alphas != 0]), len(model.alphas)))
    sv_idx = []
    for idx, alpha in enumerate(model.alphas):
        if alpha != 0:
            print('index = {}, alpha = {:.3f}, predict y={:.3f}'\
                .format(idx, alpha, train_y[idx]))
            sv_idx.append(idx)


    print(f'bias = {model.b}')
    print('training data error rate = {}'.format(len(Y[Y * train_y < 0])/len(Y)))

    norm = np.linalg.norm(model.w)
    margin = 1/norm
    print(f'Margin for C={C} is {margin}')

    # Scatter plot for positive and negative classes
    plt.scatter(X[Y == 1, 0], X[Y == 1, 1], color='r', alpha=0.55, label='Positive class')
    plt.scatter(X[Y == -1, 0], X[Y == -1, 1], color='b', alpha=0.55, label='Negative class')

    resolution = 50
    dx = np.linspace(X[:, 0].min(), X[:, 0].max(), resolution)
    dy = np.linspace(X[:, 1].min(), X[:, 1].max(), resolution)
    dx, dy = np.meshgrid(dx, dy)
    plot_x = np.c_[dx.flatten(), dy.flatten()]

    dz = model.predict(scaler.transform(plot_x))
    dz = dz.reshape(dx.shape)

    plt.contour(dx, dy, dz, alpha=1, colors=('b', 'k', 'r'), \
                levels=(-1, 0, 1), linestyles = ('--', '-', '--'))

    label_cnt = 0
    for i in sv_idx:
        if label_cnt == 0:
            plt.scatter(X[i, 0], X[i, 1], marker='*', color='k', \
                        s=150, alpha=.3, label='Support vector')
            label_cnt += 1
            continue

        plt.scatter(X[i, 0], X[i, 1], marker='*', color='k', alpha=.3, s=150)

    plt.title(f"Soft SVM (C={C})")
    plt.legend()
    plt.show()

if __name__ == '__main__':
    
    X=np.load("P2_data/data.npz")["x"]
    Y=np.load("P2_data/data.npz")["y"]

    plot(X, Y, C=1)
    plot(X, Y, C=10)
    plot(X, Y, C=100)