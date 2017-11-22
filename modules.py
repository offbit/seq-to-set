import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

PADDING_IDX = 0


def weight_init(m):
    if isinstance(m, nn.Conv1d):
        nn.init.kaiming_uniform(m.weight, mode='fan_in')
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform(m.weight, gain=nn.init.calculate_gain('relu'))


class MemoryAttention(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(MemoryAttention, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.att_conv = nn.Conv1d(in_dim, out_dim, 1)
        self.scale = Variable(torch.FloatTensor([self.in_dim]).cuda())
        weight_init(self.att_conv)

    def forward(self, key, query):
        # batch_size = key.size(0)
        query = query.unsqueeze(1)

        key = key.transpose(1, 2)
        val = self.att_conv(key)
        val = val.transpose(1, 2)

        attention = F.softmax(torch.bmm(query, key) / torch.sqrt(self.scale))
        attention = attention.transpose(1, 2)
        return (attention * val).sum(1)


class NTM(nn.Module):
    def __init__(self, memory_dim):
        super(NTM, self).__init__()
        self.memory_dim = memory_dim
        self.lstm_units = memory_dim
        self.input = memory_dim * 2
        self.lstm = nn.LSTMCell(self.input, self.lstm_units)
        self.attend = MemoryAttention(self.memory_dim, self.memory_dim)

    def forward(self, x):
        batch_size = x.size(0)
        steps = x.size(1)
        hx = Variable(torch.randn(batch_size, self.lstm_units).cuda())
        cx = Variable(torch.randn(batch_size, self.lstm_units).cuda())
        q0 = Variable(torch.randn(batch_size, self.lstm_units).cuda())
        r0 = x.mean(dim=1)
        in0 = torch.cat((q0, r0), dim=-1)
        q, cell = self.lstm(in0, (hx, cx))
        r = self.attend(x, q)
        # outputs = []
        for i in range(steps):
            in_x = torch.cat((q, r), dim=-1)
            q, cell = self.lstm(in_x, (q, cell))
            r = self.attend(x, q)
        return torch.cat((q, r), dim=-1)
