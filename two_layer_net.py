import sys
import os
sys.path.append(os.pardir)
import numpy as np
from common.gradient import numerical_gradient
from common.functions import *


# 入力層、隠れ層、出力層からなるネットワーク
class TwoLayerNet:

    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        # 重みの初期化:ディクショナリを仕様
        self.params = {}
        self.params['W1'] = weight_init_std *\
            np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)  # バイアスは0で初期化
        self.params['W2'] = weight_init_std *\
            np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)  # バイアスは0で初期化

    def predict(self, x):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']

        a1 = np.dot(x, W1)+b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2)+b2
        y = softmax(a2)

        return y

    # 損失関数の値を返す
    # x:入力データ,t:教師データ(答え)
    def loss(self, x, t):
        y = self.predict(x)

        return cross_entropy_error(y, t)

    # 認識精度を求める
    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)

        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    # 重みパラメータに対する勾配を求める
    def numerical_gradient(self, x, t):
        # ラムダ式
        def loss_W(W): return self.loss(x, t)

        grads = {}
        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])

        return grads
