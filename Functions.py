import numpy as np
import math

def save_theta(theta1, theta2):
    f1 = open('theta1.txt', 'w')
    np.savetxt(f1, theta1)
    f1.close()
    f2 = open('theta2.txt', 'w')
    np.savetxt(f2, theta2)
    f2.close()


#+normalization
def sigmoid(x):
    row, col = np.shape(x)
    tmp = np.zeros((row, col))
    for i in range(row):
        for j in range(col):
            try:
                tmp[i][j] = 1/(1 + math.exp(-1*x[i][j]))
            except OverflowError:
                tmp[i][j] = 0
    return tmp

def log(x):
    row, col = np.shape(x)
    tmp = np.zeros((row, col))
    for i in range(row):
        for j in range(col):
            tmp[i][j] = math.log(x[i][j], math.exp(1))
    return tmp

def nnCostFunction(Theta1, Theta2, InputLayerSize, HiddenLayerSize, NumLabels, X, Y, lamb):
    J = 0
    m = len(X)
    Theta2_grad = 0
    Theta1_grad = 0
    '''
    for i in range(m):
        tmp = np.reshape(X[i,:], (1, 785))
        tmp = sigmoid(np.dot(Theta1, np.transpose(tmp)))
        tmp = np.reshape(np.insert(tmp, 0, 1), (1, 26))
        tmp = sigmoid(np.dot(tmp, np.transpose(Theta2)))
        h = np.transpose(tmp)
        Ytmp = np.reshape(Y[:, i], (1, 10))
        J = J - float(np.dot(Ytmp, log(h)))
        J = J - float(np.dot(np.ones((1, 10)) - Ytmp, log(np.ones((10, 1)) - h)))
    J /= m
    '''
    for t in range(m):
        tmp = np.reshape(X[t, :], (1, InputLayerSize))
        z2 = np.dot(Theta1, np.transpose(tmp))
        a2 = sigmoid(z2)
        a2 = np.reshape(np.insert(a2, 0, 1), (1, HiddenLayerSize + 1))
        z3 = np.dot(a2, np.transpose(Theta2))
        a3 = sigmoid(z3)
        h = np.transpose(a3)
        Ytmp = np.reshape(Y[:, t], (1, NumLabels))
        J = J - float(np.dot(Ytmp, log(h)))
        J = J - float(np.dot(np.ones((1, NumLabels)) - Ytmp, log(np.ones((NumLabels, 1)) - h)))
        del3 = np.transpose(a3) - np.reshape(Y[:, t], (NumLabels, 1))
        del2 = np.dot(np.transpose(Theta2), del3)
        del2 = del2[1:]
        del2 = del2 * sigmoid(z2)
        Theta2_grad += np.dot(del3, a2)
        Theta1_grad += np.dot(del2, tmp)

    J /= m
    Theta1_grad = Theta1_grad / m
    Theta2_grad = Theta2_grad / m
    sum = 0
    sum += np.sum(Theta1**2) + np.sum(Theta2**2)
    J = J + sum*lamb/(2*m)
    k = lamb / m
    Theta1_grad = Theta1_grad + Theta1*k
    Theta2_grad = Theta2_grad + Theta2*k
    return J, Theta1_grad, Theta2_grad


def train_network(Theta1, Theta2, InputLayerSize, HiddenLayerSize, NumLabels, X, y, lamb, maxIter):
    Y = np.zeros((NumLabels, len(y)))
    for i in range(len(y)):
        for numb in range(NumLabels):
            if (y[i] == numb):
                Y[numb][i] = 1
                break;
    Jmin = 10000
    for i in range(maxIter):
        print("iteration number:" + str(i))
        J, grad1, grad2 = nnCostFunction(Theta1, Theta2, InputLayerSize, HiddenLayerSize, NumLabels, X, Y, lamb)
        print("cost: " + str(J))
        if Jmin > J:
            Jmin = J
            save_theta(Theta1, Theta2)
            print("new Theta")
        Theta1 = Theta1 - 2*grad1
        Theta2 = Theta2 - 2*grad2
    return Theta1, Theta2


def predict(Theta1, Theta2, X):
    tmp = np.reshape(X, (1, 785))
    z2 = np.dot(Theta1, np.transpose(tmp))
    a2 = sigmoid(z2)
    a2 = np.reshape(np.insert(a2, 0, 1), (1, np.shape(Theta2)[1]))
    z3 = np.dot(a2, np.transpose(Theta2))
    a3 = sigmoid(z3)
    h = np.transpose(a3)
    return int(np.argmax(h, axis=0))