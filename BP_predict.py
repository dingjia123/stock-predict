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
file_name = config_BP.file_name
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



dirs = os.path.join('./model',file_name.split('.')[0])
print('dirs',dirs)

for days in ['30day','15day','5day']:
    print('~~~~~~~~~~~~~~~~~~~`')
    end_res = []
    path_day = os.path.join(dirs,days)
    print('加载模型路径是：',path_day)

    TIMESTEPS = int(days.split('day')[0])
    model = get_model(TIMESTEPS*3)
    for i in range(5):
        # 加载
        path_model = os.path.join(path_day,str(i) + '.pth')
        model.load_state_dict(torch.load(path_model))


        test_features = np.concatenate((shuju_high[-TIMESTEPS:].reshape(1,-1),shuju_low[-TIMESTEPS:].reshape(1,-1),
                                       shuju_close[-TIMESTEPS:].reshape(1,-1)),axis=1)
        test_features = torch.from_numpy(test_features).type(torch.FloatTensor).reshape(1, -1)
        print(test_features.shape)
        preds = predict(model, test_features, False)
        print('第{}个模型预测结果是：{}'.format(i,preds[0]))
        # end_res.append(preds[0][0].item())
    #
    # res = np.array(end_res).mean()
    # print('days------',days,judge,res)
