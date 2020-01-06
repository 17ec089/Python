# TwoLayerNetの学習

import sys
import os
sys.path.append(os.pardir)
from two_layer_net import TwoLayerNet
from dataset.mnist import load_mnist
import numpy as np


# 学習データとテストデータの読み込み
(x_train, t_train), (x_test, t_test) =\
    load_mnist(normalize=True, one_hot_label=True)

# ハイパーパラメタ
iters_num = 10000
train_size = x_train.shape[0]  # 60000
batch_size = 100
learning_rate = 0.1

train_loss_list = []
train_acc_list = []
test_acc_list = []
# 1エポック当たりの繰り返し数
iter_per_epoch = max(train_size/batch_size, 1)

# 入力層:784画素 隠れ層:50個 出力層:0~9のうちのどれか
network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

# 1万回勾配の更新を行う
for i in range(iters_num):
    # ミニバッチの取得
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    # 勾配の計算
    grad = network.numerical_gradient(x_batch, t_batch)

    # パラメタの更新
    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate*grad[key]

    # 学習経過の記録(損失関数の計算値の記録)
    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)

    # 1エポックごとに認識精度を計算
    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)

        print("train acc, test acc | " + str(train_acc)+", " + str(test_acc))
