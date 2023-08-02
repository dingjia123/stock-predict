# -- coding: utf-8 --
import numpy as np
import os
import pandas as pd
import random

import torch
from torch.utils.data import DataLoader
# import matplotlib as mpl
# mpl.use('Agg')
from matplotlib import pyplot as plt
import re
from model import FeedBackDataset,LstmModel
import shutil
import config_lstm


def set_seed(seed):
    """
    设置随机种子
    :param seed:
    :return:
    """
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
set_seed(123)


file_name = config_lstm.file_name
dirs = os.path.join('./model', file_name.split('.')[0])
print('dirs--------', dirs)
if not os.path.exists(dirs):
    os.makedirs(dirs)
else:
    shutil.rmtree(dirs)
    os.makedirs(dirs)


lr = config_lstm.lr
device = torch.device("cpu")
path = r'E:\stockPredict\gupiao\datas'
shuju = pd.read_excel(os.path.join(path,file_name))
print('加载的数据地址：',os.path.join(path,file_name))

pattern = re.compile(r' ')
cols = [re.sub(pattern,'',col) for col in list(shuju.columns)]
shuju.columns = cols

save_cols = ['时间','开盘','最高','最低','收盘']
shuju = shuju[save_cols]

data = shuju[['开盘','最高','最低','收盘']]
data_array = data.values
print(data_array[-5:])


TIMESTEPS = 5          #循环网络中训练序列的长度

#将数据切分，做成训练集和测试集
def generate_data(seq,TIMESTEPS):
    x = []
    y = []
    for i in range(len(seq)-TIMESTEPS):
        x.append([seq[i:i+TIMESTEPS]])
        y.append([seq[i+TIMESTEPS]])
    return np.array(x,dtype=np.float32),np.array(y,dtype=np.float32)



def train_model(LstmModel,epochs,model_path):
    optimizer = torch.optim.Adam(LstmModel.parameters(), lr=lr)
    LstmModel.zero_grad()

    best_loss = 1000000000.0
    best_epcoh = 0
    for e in range(epochs):
        # 训练模型
        running_loss = 0
        count = 0
        for data in train_loader:
            LstmModel.train()
            x, y = data['x'], data['y'].reshape(data['x'].shape[0], 3)
            x = torch.tensor(x).to(device)
            y = torch.tensor(y).to(device)
            # todo: 前向传播
            loss, y_pred = LstmModel(x, y)
            # todo: 反向传播，更新参数
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss
            count = count + 1
        loss_epoch = running_loss /count
        if e % 50 == 0:
            print('{} epoch的loss数值是{}'.format(e,loss_epoch.item()))
        if best_loss > loss_epoch:
            best_loss = loss_epoch
            best_epcoh = e
            torch.save(LstmModel.state_dict(), model_path)
            # print('目前最好的模型分数是{}， {}'.format(best_loss,e))

    # torch.save(LstmModel.state_dict(), model_path)
    print('目前最好的模型分数是{}， {}'.format(best_loss,best_epcoh))
    print('保存成功，地址{}'.format(model_path))



for days in ['30day', '15day', '5day']:
    path_day = os.path.join(dirs, days)
    print('每部分的天数')
    print(path_day)
    TIMESTEPS = int(days.split('day')[0])
    print('TIMESTEPS--------TIMESTEPS', TIMESTEPS)
    if not os.path.exists(path_day):
        os.makedirs(path_day)
    else:
        shutil.rmtree(path_day)
        os.makedirs(path_day)

    train_x, train_y = generate_data(data_array, TIMESTEPS)

    train_x = train_x.reshape(-1, TIMESTEPS, 4)
    train_y = train_y.reshape(-1, 4)
    train_y = train_y[:, 1:]
    print(train_x.shape)
    print(train_y.shape)

    train_dataset = FeedBackDataset(train_x, train_y, mode='train')
    train_loader = DataLoader(train_dataset, batch_size=64, num_workers=0, shuffle=True)

    for i in range(5):
        model_path = os.path.join(path_day, str(i) + '.pth')
        print('保存路径是:', model_path)
        model = None
        model = LstmModel(4, 16, 64)
        model.to(device)
        train_model(model,epochs=500,model_path=model_path)

        # # 保存
        # torch.save(LstmModel.state_dict(), model_path)






