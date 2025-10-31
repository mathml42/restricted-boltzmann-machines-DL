import numpy as np
from keras.datasets import fashion_mnist

# Load dataset
def load_data():
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    class_type = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    x_train = np.array(x_train.reshape(x_train.shape[0], 784,1))
    x_test = np.array(x_test.reshape(x_test.shape[0], 784, 1))
    x_train = (x_train > 126) * 1
    x_test = (x_test > 126) * 1

    x_val = x_train[-15000:]
    x_train = x_train[0:45000]

    Y_train = np.zeros([len(y_train), 10, 1])
    Y_test = np.zeros([len(y_test), 10, 1])

    for i in range(len(y_train)):
        y = np.zeros([10,1])
        y[y_train[i]] = 1.0
        Y_train[i] = y
    
    for i in range(len(y_test)):
        y = np.zeros([10,1])
        y[y_test[i]] = 1.0
        Y_test[i] = y
    
    Y_val = Y_train[-15000:]
    Y_train = Y_train[0:45000]

    return x_train, Y_train, x_val, Y_val, x_test, Y_test

if __name__=="__main__":
    load_data()