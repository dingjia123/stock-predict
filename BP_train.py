# -- coding: utf-8 --
import numpy as np
import os
import pandas as pd
import re
from pandas import DataFrame,Series

import torch
from torch import nn
from torch.autograd import Variable
from utils import get_rmse_log,get_mse_log
from model_BP import train_model
import config_BP

import shutil
import os

# 可以调整的超参
batch_size = config_BP.batch_size
epochs = config_BP.epochs
use_gpu = config_BP.use_gpu
lr = config_BP.lr
weight_decay = config_BP.weight_decay
TIMESTEPS = config_BP.TIMESTEPS          #循环网络中训练序列的长度
file_name = config_BP.file_name

judge = config_BP.judge



#将数据切分，做成训练集和测试集
def generate_data(seq,TIMESTEPS):
    x = []
    y = []
    for i in range(len(seq)-TIMESTEPS):
        x.append([seq[i:i+TIMESTEPS]])
        y.append([seq[i+TIMESTEPS]])
    return np.array(x,dtype=np.float32),np.array(y,dtype=np.float32)


def get_model(len_pros):
    # todo: 使用 nn.Sequential 来构造多层神经网络，注意第一层的输入
    model = nn.Sequential(
        nn.Linear(len_pros, 128), nn.ReLU(),
        nn.Linear(128, 64),nn.ReLU(),
        nn.Linear(64,16),nn.GELU(),
        nn.Linear(16, 3))
    return model



def predict(model, feature,  use_gpu):
    model.eval()
    if use_gpu:
        feature = feature.cuda()
    with torch.no_grad():
        feature = feature
    pred = model(feature)
    return pred



# 数据获取
path = r'E:\stockPredict\gupiao\datas'
shuju = pd.read_excel(os.path.join(path,file_name))

pattern = re.compile(r' ')
cols = [re.sub(pattern,'',col) for col in list(shuju.columns)]
shuju.columns = cols

save_cols = ['时间','开盘','最高','最低','收盘']
shuju = shuju[save_cols]
print('数据数据：')
print(shuju[:2])





shuju_low = np.array(shuju['最低'])
shuju_high = np.array(shuju['最高'])
shuju_close = np.array(shuju['收盘'])

print(len(shuju_close))
# shuju_ = shuju_[:-50]
# print(len(shuju_))
print(shuju_close[-5:])

dirs = os.path.join('./model', file_name.split('.')[0])
print('dirs', dirs)
if not os.path.exists(dirs):
    os.makedirs(dirs)
else:
    shutil.rmtree(dirs)
    os.makedirs(dirs)

for days in ['30day', '15day', '5day']:
    path_day = os.path.join(dirs, days)
    print('每部分的天数')
    print(path_day)
    if not os.path.exists(path_day):
        os.makedirs(path_day)
    else:
        shutil.rmtree(path_day)
        os.makedirs(path_day)

    TIMESTEPS = int(days.split('day')[0])
    print('TIMESTEPS--------TIMESTEPS', TIMESTEPS)
    train_x_low, train_y_low = generate_data(shuju_low, TIMESTEPS)
    train_x_high, train_y_high = generate_data(shuju_high, TIMESTEPS)
    train_x_close, train_y_close = generate_data(shuju_close, TIMESTEPS)

    train_x_low, train_y_low = train_x_low.reshape(len(train_x_low), -1), train_y_low.reshape(len(train_y_low), -1)
    train_x_high, train_y_high = train_x_high.reshape(len(train_x_high), -1), train_y_high.reshape(len(train_y_high),
                                                                                                   -1)
    train_x_close, train_y_close = train_x_close.reshape(len(train_x_close), -1), train_y_close.reshape(
        len(train_y_close), -1)

    #     print('train_x_low.shape',train_x_low.shape)
    #     print('train_y_low.shape', train_y_low.shape)

    train_x = np.concatenate((train_x_high, train_x_low, train_x_close), axis=1)
    train_y = np.concatenate((train_y_high, train_y_low, train_y_close), axis=1)
    print(train_x.shape)
    print(train_y.shape)
    #     print(train_y[-5:])

    # 将数据放到Torch中
    train_features = torch.from_numpy(train_x).type(torch.FloatTensor)
    train_labels = torch.from_numpy(train_y).type(torch.FloatTensor)

    for i in range(5):
        model = get_model(train_features.shape[1])

        train_model(model, train_features, train_labels, None, None, epochs, lr,
                    weight_decay, batch_size, use_gpu)
        print('第几个模型：', i)
        # print(list(model[0].parameters())[0][:2])
        # 保存
        model_path = os.path.join(path_day,str(i) + '.pth')
        print('保存路径是:',model_path)
        torch.save(model.state_dict(),model_path )
