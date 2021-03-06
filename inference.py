import sys
import os
import pickle
sys.path.append(os.pardir)
from common.functions import sigmoid, softmax
import numpy as np
from dataset.mnist import load_mnist


# テストデータを返す
def get_data():
    # tはラベル
    (x_train, t_train), (x_test, t_test) = \
        load_mnist(normalize=True, flatten=True, one_hot_label=False)
    return x_test, t_test


# 学習済み重みパラメタを返す
def init_network():
    with open("sample_weight.pkl", 'rb') as f:
        network = pickle.load(f)

    return network


# 分類を行う x:画像データ
def predict(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, W1)+b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2)+b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3)+b3
    y = softmax(a3)

    return y


'''
#バッチを用いない
x,t=get_data()
network = init_network()
accuracy_cnt = 0

#1万枚
for i in range(len(x)):
    y=predict(network,x[i])
    p=np.argmax(y)
    if p==t[i]:
        accuracy_cnt+=1

#１万枚のうち、何枚正解したか
print("認識精度"+str(float(accuracy_cnt)/len(x)))
'''

# バッチを使用
x, t = get_data()
network = init_network()

batch_size = 100
accuracy_cnt = 0

for i in range(0, len(x), batch_size):
    x_batch = x[i:i+batch_size]
    y_batch = predict(network, x_batch)
    p = np.argmax(y_batch, axis=1)
    accuracy_cnt += np.sum(p == t[i:i+batch_size])

print("認識精度"+str(float(accuracy_cnt)/len(x)))
