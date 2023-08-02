# -- coding: utf-8 --

import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader,Dataset
import os

# 读取数据函数
class FeedBackDataset(Dataset):
    def __init__(self, x,y=None,mode='train'):
        self.x = x
        if mode == 'train':
            self.y = y
        else:
            self.y = None

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        data = {'x':self.x[index]}
        if self.y is not None:
            data['y'] = self.y[index].reshape(1,-1)
        return data


class LstmModel(nn.Module):
    def __init__(self, hidden_size, rnn_dim, mid_linear_dims, dropout_prob=0.1, **kwargs):
        super(LstmModel, self).__init__()
        '''
        Args:
            hidden_size: 输入数值的最后维度
            rnn_dim: lstm输出的维度
            mid_linear_dims: 线性层
            dropout_prob:
            **kwargs:
        '''
        self.hidden_size = hidden_size
        self.rnn_dim = rnn_dim
        self.mid_linear_dims = mid_linear_dims
        self.dropout_prob = dropout_prob
        self.birnn = nn.LSTM(self.hidden_size, self.rnn_dim, num_layers=1, bidirectional=False,
                             batch_first=True)

        self.mid_linear = nn.Sequential(
            nn.Linear(self.rnn_dim, self.mid_linear_dims),
            nn.ReLU(),
            nn.Dropout(self.dropout_prob)
        )

        self.classifier = nn.Linear(self.mid_linear_dims, 3)
        init_blocks = [self.mid_linear, self.classifier]
        # 损失函数
        self.criterion = nn.MSELoss()

        self._init_weights(init_blocks)

    def _init_weights(self, blocks, **kwargs):
        """
        参数初始化，将 Linear 进行初始化
        """
        for block in blocks:
            for module in block.modules():
                if isinstance(module, nn.Linear):
                    nn.init.zeros_(module.bias)

    def forward(self, x, y=None, ):
        birnn_outputs,_ = self.birnn(x)
        seq_out = birnn_outputs[:, -1, :]
        seq_out = self.mid_linear(seq_out)
        logits = self.classifier(seq_out)
        out = (logits,)
        if y is not None:
            loss = self.criterion(logits, y.float())
            out = (loss,) + out
        return out

def save_model(opt, model, global_step):
    output_dir = os.path.join(opt.output_dir, 'checkpoint-{}'.format(global_step))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # take care of model distributed / parallel training
    model_to_save = (
        model.module if hasattr(model, "module") else model
    )
    torch.save(model_to_save.state_dict(), os.path.join(output_dir, 'model.pt'))