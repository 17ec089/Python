# ReLUレイヤ
class Relu:
    def __init__(self):
        self.mask = None

    def forword(self, x):
        self.mask = (x <= 0)  # 0以下ならばTrue
        out = x.copy()
        out[self.mask] = 0  # Trueを0にする

        return out

    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout

        return dx


# Sigmoidレイヤ
class Sigmoid:
    def __init__(self):
        self.out = None

    def forword(self, x):
        out = 1/(1+np.exp(-x))
        self.out = out

        return out

    def backward(self, dout):
        dx = dout*(1.0-self.out)*self.out

        return dx


#Affineレイヤ
class Affine:
    def __init__(self, W, b):
        self.W=W #重み
        self.b=b #バイアス
        self.x=None
        self.dW=None
        self.db=None

    def forword(self, x):
        self.x=x
        out=np.dot(x,self.W)+self.b
        
        return out

    def backward(self,dout):
        dx=np.dot(dout,self.W.T)
        self.dW=np.dot(self.x.T,dout)
        self.db=np.sum(dout,axis=0)

        return dx
    

    #SoftmaxとLossの混合レイヤ
    class SoftmaxWithLoss:
        def __init__(self):
            self.loss=None #損失
            self.y=None #softmaxの出力
            self.t=None #教師データ,one-hot表現
        
        def forword(self,x,t):
            self.t=t
            self.y=softmax(x)
            self.loss=cross_entropy_error(self.y,self.t)

            return self.loss
        
        def backward(self,dout=1):
            batch_size=self.t.shape[0]
            dx=(self.y-self.t)/batch_size

            return dx
