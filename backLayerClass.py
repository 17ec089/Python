import sys,os
sys.path.append(os.pardir)
import numpy as np
from common.layers import *
from common.gradient import numerical_gradient
from collections import OrderedDict

class TwoLayerNet:

    def __init__(self, input_size, hidden_size, output_size,weight_init_std=0.01):

        #重みの初期化
        self.params={}
        self.params['W1']=weight_init_std*\
            np.random.randn(input_size,hidden_size)
        self.params['b1']=np.zeros(hidden_size)
        self.params['W2']=weight_init_std*\
            np.random.randn(output_size,output_size)
        self.params['b2']=np.zeros(output_size)

        #レイヤの作成
        self.layers=OrderedDict()
        self.layers['Affine1']=\
            Affine(self.params['W1'],self.params['b1'])
        self.layers['Relu1']=Relu()
        self.layers['Affine2']=\
            Affine(self.params['W2'],self.params['b2'])

        self.lastLayer=SoftmaxWithLoss()

    def predict(self,x):
        for layer in self.layers.values():
            x=layer.forward(x) #順伝搬

        return x #Affine2からの出力

    #x:入力データ, t:教師データ
    def loss(self, x, t):
        y =self.predict(x)
        return self.lastLayer.forward(y,t) #順伝搬

    def accuracy(self, x, t):
        y=self.predict(x)
        y=np.argmax(y, axis=1) #行方向
        if t.ndim != 1:
            t=np.argmax(t, axis=1)

        accuracy=np.sum(y==t)/float(x.shape[0])
        return accuracy

    def numerical_gradient(self,x,t):
        loss_W=lambda W: self.loss(x,t)

        grads={}
        grads['W1']=numerical_gradient(loss_W,self.params['W1'])
        grads['b1']=numerical_gradient(loss_W,self.params['b1'])
        grads['W2']=numerical_gradient(loss_W,self.params['W2'])
        grads['b2']=numerical_gradient(loss_W,self.params['b2'])

        return grads

    #誤差逆伝搬法により勾配を求める
    def gradient(self, x, t):
        #forward
        self.loss(x, t)

        #backward
        dout = 1
        dout = self.lastLayer.backward(dout) #softmaxWithLossの逆伝搬

        layers=list(self.layers.values()) #リスト化
        layers.reverse() #逆順に
        for layer in layers:
            dout=layer.backward(dout)

        #設定
        grads={}
        grads['W1']=self.layers['Affine1'].dW
        grads['b1']=self.layers['Affine1'].db
        grads['W2']=self.layers['Affine2'].dW
        grads['b2']=self.layers['Affine2'].db

        return grads

