import numpy as np
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

def myLDA(x, y, k):
    '''
    x为数据集, y为target, k为目标维数
    '''
    target_list = list(set(y))

    x_cls = {}

    for target in target_list:
        x1 = np.array([x[i] for i in range(len(x)) if y[i] == target])
        x_cls[target] = x1

    mu = np.mean(x, axis=0)
    mu_cls = {}

    for target in target_list:
        mu1 = np.mean(x_cls[target], axis=0)
        mu_cls[target] = mu1

    #St = np.dot((x - mu).T, x - mu)
    # 计算类内散度矩阵
    Sw = np.zeros((len(mu), len(mu)))  
    for i in target_list:
        Sw += np.dot((x_cls[i] - mu_cls[i]).T,
                     x_cls[i] - mu_cls[i])

    # Sb=St-Sw
    # 计算类内散度矩阵
    Sb = np.zeros((len(mu), len(mu)))  
    for i in target_list:
        Sb += len(x_cls[i]) * np.dot((mu_cls[i] - mu).reshape(
            (len(mu), 1)), (mu_cls[i] - mu).reshape((1, len(mu))))

    # 计算Sw-1*Sb的特征值和特征矩阵
    eig_vals, eig_vecs = np.linalg.eig(
        np.linalg.inv(Sw).dot(Sb))  

    # 提取前k个特征向量
    sorted_indices = np.argsort(eig_vals)
    topk_eig_vecs = eig_vecs[:, sorted_indices[:-k - 1:-1]]  
    return topk_eig_vecs


if '__main__' == __name__:

    iris = load_iris()
    x = iris.data
    y = iris.target

    lda = myLDA(x, y, 2)
    x_new = np.dot(x, lda)
    plt.figure(1)
    plt.scatter(x_new[:, 0], x_new[:, 1], marker='o', c=y)
    plt.show()