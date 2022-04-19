'''In the name of God'''

# imports:
import numpy as np
import pandas as pd
# load data:
df = np.array(pd.read_csv(r'C:\Users\M\Desktop\DataSets\Mnist/train.csv'))
X_train = df[:,1:]
Y_train = df[:,0]
# make weights and bayas:
def init():
    w1 = np.random.randn(32,784) -0.5
    b1 = np.random.randn(32,1) -0.5
    w2 = np.random.randn(10,32) -0.5
    b2 = np.random.randn(10,1) -0.5
    return w1,b1,w2,b2
# ReLU activation:
def ReLU(Z):
    return np.maximum(Z,0)
# softmax activation:
def softmax(predicted):
    numpy_Es = sum([np.e**i for i in predicted])
    predicted = [(np.e**i)/numpy_Es for i in predicted]
    return predicted , np.argmax(predicted)
# dot the weights asd bayas to X_train:
def forward(w1,b1,w2,b2,X):
    Z1 = w1.dot(X) + b1
    A1 = ReLU(Z1)
    Z2 = w2.dot(A1) + b2
    A2 = softmax(Z2)
    return Z1,A1,Z2,A2
# ont_hot:
def one_hot(y):
    o = np.zeros((y.size,y.max()+1))
    o[np.arange(y.size),y] = 1
    return o.T
# deriv ReLU:
def deriv_ReLU(z):
    return z > 0
# back prop:
def back(z1,a1,z2,a2,w2,x,y):
    m = y.size
    one_hot_y =  one_hot(y)
    dz2 = a2 - one_hot_y
    dw2 = 1 / m * dz2.dot(a1.T)
    db2 = 1 / m * np.sum(dz2,2)
    dz1 = w2.T.dot(dz2) * deriv_ReLU(z1)
    dw1 = 1 / m * dz1.dot(x.T)
    db1 = 1/ m * np.sum(dz1,2)
    return dw1 , db1 , dw2 , db2
# update parametrs:
def upadte(w1,b1,w2,b2,dw1,db1,dw2,db2,alpha):
    w1 = w1 - dw1 * alpha
    b1 = b1 - db1 * alpha
    w2 = w2 - dw2 * alpha
    b2 = b2 - db2 * alpha
    return w1,b1,w2,b2
# prediction:
def predict(a2):
    return np.argmax(a2,0)
# accuracy:
def accuracy(predicted,y):
    print(predicted,y)
    return np.sum(predicted==y) / y.size
# gradient deisece:
def Grad(x , y , iters , alpha):
    w1 , b1 , w2 , b2 = init()
    for i in range(iters):
        z1 , a1 , z2 , a2 = forward(w1,b1,w2,b2,x)
        dw1 , db1 , dw2 , db2 = back(z1,a1,z2,a2,w2,x,y)
        w1 , b1 , w2 , b2 = upadte(w1,b1,w2,b2,dw1,db1,dw2,db2,alpha)
        if i%10 == 0:
            print('Accuracy:',accuracy(predict(a2),y))
    return w1,b1,w2,
# train model:
w1 ,b1 ,w2 ,b1 = Grad(X_train.T,Y_train,100,0.1)
print(w1,b1,w2,b2)
