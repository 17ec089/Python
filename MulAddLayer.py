# 乗算レイヤ
class MulLayer:
    def __init__(self):
        self.x = None
        self.y = None

    def forward(self, x, y):
        self.x = x  # backで使用
        self.y = y  # 上記同様
        out = x*y

        return out

    def backward(self, dout):
        dx = dout*self.y
        dy = dout*self.x

        return dx, dy


# 加算レイヤ
class AddLayer:
    def __init__(self):
        pass

    def forward(self, x, y):
        out = x+y
        return out

    def backward(self, dout):
        dx = dout*1
        dy = dout*1
        return dx, dy
