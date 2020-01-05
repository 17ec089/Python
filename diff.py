# 中心差分
def numerical_diff(f, x):
    h = 1e-4
    return (f(x+h)-f(x-h))/(2*h)


# 2変数の関数
def function_2(x):
    return x[0]**2+x[1]**2


# x0=3,x1=4の時のx0に対する偏微分
def func_tmp1(x0):
    return x0*x0+4.0**2.0

numerical_diff(func_tmp1, 3.0)


# x0=3,x1=4の時のx1に対する偏微分
def func_tmp1(x1):
    return 3.0**2.0+x1*x1

numerical_diff(func_tmp1, 4.0)


# 勾配を求める
def numerical_grad(f, x):
    h = 1e-4
    grad = np.zeros_like(x)

    for idx in range(x.size):
        tmp = x[idx]

        x[idx] = tmp+h
        fxh1 = f(x)

        x[idx] = tmp-h
        fxh2 = f(x)

        grad[idx] = (fxh1-fxh2)/(2*h)
        x[idx] = tmp

    return grad

# 勾配降下法
def grad_descent(f, init_x, lr=0.01, step_num=100):
    x = init_x

    for i in range(step_num):
        grad = numerical_grad(f, x)
        x -= lr*grad

    return x
