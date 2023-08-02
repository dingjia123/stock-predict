# -- coding: utf-8 --
import numpy as np
import os
import pandas as pd
import random
import torch
# import matplotlib as mpl
# mpl.use('Agg')
from matplotlib import pyplot as plt
import re
from model import FeedBackDataset,LstmModel
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

lr = 0.001
file_name = config_lstm.file_name
dirs = os.path.join('./model', file_name.split('.')[0])
print('dirs--------', dirs)



device = torch.device("cpu")
path = r'E:\stockPredict\gupiao\datas'
shuju = pd.read_excel(os.path.join(path,file_name))

pattern = re.compile(r' ')
cols = [re.sub(pattern,'',col) for col in list(shuju.columns)]
shuju.columns = cols

save_cols = ['时间','开盘','最高','最低','收盘']
shuju = shuju[save_cols]

data = shuju[['开盘','最高','最低','收盘']]
data_array = data.values


TIMESTEPS = 5          #循环网络中训练序列的长度

print(data_array.shape)
print(data_array[-TIMESTEPS:])


for days in ['30day','15day','5day']:
    print('~~~~~~~~~~~~~~~~~~~`')
    end_res = []
    path_day = os.path.join(dirs,days)
    print('加载模型路径是：',path_day)

    TIMESTEPS = int(days.split('day')[0])
    predict_data = data_array[-TIMESTEPS:]
    predict_data = predict_data.reshape(-1, TIMESTEPS, 4)

    model = None
    model = LstmModel(4, 16, 64)

    res = []
    for i in range(5):
        # 加载
        path_model = os.path.join(path_day,str(i) + '.pth')
        model.load_state_dict(torch.load(path_model))
        model.to(device)
        # print('path_model===',path_model)

        model.eval()
        x = predict_data
        x = torch.tensor(x,dtype=torch.float32).to(device)
        # todo: 前向传播
        pred = model(x,y=None)
        print(pred[0])
        res.append(pred[0].detach().numpy())
    res = np.array(res).reshape(-1,3)
    mean_res = np.mean(res,axis=0)
    # print(mean_res)

    print('{}的最高价均值:{},最低价均值：{}，收盘价均值：{}'.format(days,mean_res[0],mean_res[1],mean_res[2]))
