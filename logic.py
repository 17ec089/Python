import numpy as np 

# AND
def AND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5]) # 重み
    b = -0.7 # バイアス
    y = np.sum(w*x) + b # 各要素の乗算和
    if y <= 0:
        return 0
    else:
        return 1


# NAMD
def NAND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([-0.5, -0.5])
    b = 0.7
    y = np.sum(w*x) + b
    if y <= 0:
        return 0
    else:
        return 1


# OR
def OR(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.2
    y = np.sum(w*x) + b
    if y <= 0:
        return 0
    else:
        return 1


# XOR
def XOR(x1, x2):
    s1 = NAND(x1, x2)
    s2 = OR(x1, x2)
    y = AND(s1, s2)
    return y
