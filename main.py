
import Functions
from keras.datasets import mnist
import numpy as np
from PIL import Image

def ToMnist(X):
    X = X.sum(axis=-1)
    for i in range(0, np.shape(X)[0]):
        for j in range(0, np.shape(X)[1]):
            if X[i][j] == 1020:
                X[i][j] = 0
            else:
                X[i][j] = 1
    return X


def load_theta():
    th1 = np.loadtxt('theta1.txt')
    th2 = np.loadtxt('theta2.txt')
    return th1, th2


def save_theta(theta1, theta2):
    f1 = open('theta1.txt', 'w')
    np.savetxt(f1, theta1)
    f1.close()
    f2 = open('theta2.txt', 'w')
    np.savetxt(f2, theta2)
    f2.close()


(X_train, y), (X_test, y_test) = mnist.load_data()


X = np.zeros((60000, 785))


for i in range(60000):
    X[i][0] = 1
    X[i][1:785] = X_train[i].ravel()



print("starting normalization:")
for i in range(60000):
    for j in range(1, 785):
        if X[i][j] < 160:
            X[i][j] = 0
        else:
            X[i][j] = 1
    if i % 10000 == 0:
        print(i/60000)
print("normalization finished")


input_layer_size = 785
hidden_layer_size = 400
num_labels = 10
#Theta1 = (np.random.random((hidden_layer_size, input_layer_size)) * (2 * 0.16)) - 0.16
#Theta2 = (np.random.random((num_labels, hidden_layer_size + 1)) * (2 * 0.16)) - 0.16

Theta1, Theta2 = load_theta()
lamb = 0
#J, grad1, grad2 = Functions.nnCostFunction(Theta1, Theta2, input_layer_size, hidden_layer_size, num_labels, X, y, lamb)#Theta1, Theta2 = load_theta()
Theta1, Theta2 = Functions.train_network(Theta1, Theta2, input_layer_size, hidden_layer_size, num_labels, X[:59000], y[:59000], lamb, 100)
numbCorrect = 0

for i in range(59000, 60000):
    if Functions.predict(Theta1, Theta2, X[i]) == y[i]:
        numbCorrect += 1
print("accuracy:" + str(numbCorrect/60000))
print("РАСКОММЕНТИТЬ ЛОАД!!!!!!!!")



I = np.asarray(Image.open('test.png'))
I = np.array(I)
I = ToMnist(I)
testEx = np.zeros(785)
testEx[0] = 1
testEx[1:785] = I.ravel()


print("PREDICTION: ")
print(Functions.predict(Theta1, Theta2, testEx))


