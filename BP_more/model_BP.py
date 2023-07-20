# -- coding: utf-8 --
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from utils import get_rmse_log,get_mse_log
# %matplotlib inline
# import matplotlib as mpl
# mpl.use('Agg')
from matplotlib import pyplot as plt

# 损失函数
criterion = nn.MSELoss()


# 构建一个数据的迭代器
def get_data(x, y, batch_size, shuffle):
    dataset = TensorDataset(x, y)
    return DataLoader(dataset, batch_size, shuffle=shuffle, num_workers=0)

# 模型训练
def train_model(model, x_train, y_train, x_valid, y_valid, epochs, lr, weight_decay,batch_size,use_gpu):
    metric_log = dict()
    metric_log['train_loss'] = list()
    if x_valid is not None:
        metric_log['valid_loss'] = list()

    train_data = get_data(x_train, y_train, batch_size, True)
    if x_valid is not None:
        valid_data = get_data(x_valid, y_valid, batch_size, False)
    else:
        valid_data = None

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for e in range(epochs):
        # 训练模型
        running_loss = 0
        model.train()
        for data in train_data:
            x, y = data
            if use_gpu:
                x = x.cuda()
                y = y.cuda()


            # todo: 前向传播
            y_pred = model(x)
            # todo: 计算 loss
            if len(y.shape) == 1:
                y = y.reshape(-1, 3)
            loss = criterion(y_pred, y)
            # todo: 反向传播，更新参数
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        metric_log['train_loss'].append(get_mse_log(model, x_train, y_train, use_gpu))

        # 测试模型
        if x_valid is not None:
            metric_log['valid_loss'].append(get_mse_log(model, x_valid, y_valid, use_gpu))
            print_str = 'epoch: {}, train loss: {:.3f}, valid loss: {:.3f}' \
                .format(e + 1, metric_log['train_loss'][-1], metric_log['valid_loss'][-1])
        else:
            print_str = 'epoch: {}, train loss: {:.3f}'.format(e + 1, metric_log['train_loss'][-1])
        if e % 50 == 0:
            print(print_str)

    # =======不要修改这里的内容========
    # 可视化
    # figsize = (10, 5)
    # fig = plt.figure(figsize=figsize)
    # plt.plot(metric_log['train_loss'], color='red', label='train')
    # if valid_data is not None:
    #     plt.plot(metric_log['valid_loss'], color='blue', label='valid')
    # plt.legend(loc='best')
    # plt.xlabel('epochs')
    # plt.ylabel('loss')
    # plt.show()
