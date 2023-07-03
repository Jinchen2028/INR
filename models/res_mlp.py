import torch.nn as nn
import torch

from models import register


@register('res_mlp')
class Res_mlp(nn.Module):

    def __init__(self, in_dim, out_dim, hidden_list):
        super().__init__()
        # layers = []
        hidden = hidden_list[0]
        lastv0 = in_dim
        # for hidden in hidden_list:
        #     layers.append(nn.Linear(lastv, hidden))
        #     layers.append(nn.ReLU())
        #     lastv = hidden
        # layers.append(nn.Linear(lastv, out_dim))
        # self.layers = nn.Sequential(*layers)
        self.act = nn.ReLU()
        self.layer_1 = nn.Linear(lastv0, hidden)
        lastv1 = hidden + 46
        self.layer_2 = nn.Linear(lastv1, hidden)
        self.layer_3 = nn.Linear(lastv1, hidden)
        self.layer_4 = nn.Linear(lastv1, hidden)
        self.layer_5 = nn.Linear(lastv1, out_dim)

    def forward(self, x, coord_):
        # print(coord_.shape)
        shape = x.shape[:-1]
        # print(shape)
        x_1 = self.layer_1(x.view(-1, x.shape[-1]))
        x_2 = self.act(x_1)
        x_2 = self.layer_2(torch.cat([x_2, coord_], dim=-1))
        x_3 = self.act(x_2)
        x_3 = self.layer_3(torch.cat([x_3, coord_], dim=-1))
        x_4 = self.act(x_1 + x_3) #res
        x_4 = self.layer_4(torch.cat([x_4, coord_], dim=-1))
        x_5 = self.act(x_4)
        x = self.layer_5(torch.cat([x_5, coord_], dim=-1))

        return x.view(*shape, -1)