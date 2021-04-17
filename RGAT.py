
from dgl.nn.pytorch import GATConv

import torch
import torch.nn as nn
import torch.nn.functional as F


class GATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, edge_embed_dim=50):
        super(GATLayer, self).__init__()
        # equation (1)
        self.fc = nn.Linear(in_dim, out_dim, bias=False)
        # equation (2)
        self.attn_fc = nn.Linear(2 * out_dim + edge_embed_dim, 1, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.fc.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_fc.weight, gain=gain)

    def edge_attention(self, edges):
        # edge UDF for equation (2)
        z2 = torch.cat([edges.src['z'], edges.data['y'], edges.dst['z']], dim=1)
        a = self.attn_fc(z2)
        return {'e': F.leaky_relu(a)}

    def message_func(self, edges):
        # message UDF for equation (3) & (4)
        return {'z': edges.src['z'], 'e': edges.data['e']}

    def reduce_func(self, nodes):
        # reduce UDF for equation (3) & (4)
        # equation (3)
        alpha = F.softmax(nodes.mailbox['e'], dim=1)
        # equation (4)
        h = torch.sum(alpha * nodes.mailbox['z'], dim=1)
        return {'h': h}

    def forward(self, g, h):
        # equation (1)
        z = self.fc(h)
        g.ndata['z'] = z
        # equation (2)
        g.apply_edges(self.edge_attention)
        # equation (3) & (4)
        g.update_all(self.message_func, self.reduce_func)
        nh = g.ndata.pop('h')
        return nh

class MultiHeadGATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads, merge='cat', edge_embed_dim=50):
        super(MultiHeadGATLayer, self).__init__()
        self.heads = nn.ModuleList()
        for i in range(num_heads):
            self.heads.append(GATLayer(in_dim, out_dim, edge_embed_dim))
        self.merge = merge

    def forward(self, g, h):
        head_outs = [attn_head(g, h) for attn_head in self.heads]
        if self.merge == 'cat':
            # concat on the output feature dimension (dim=1)
            return torch.cat(head_outs, dim=1)
        else:
            # merge using average
            return torch.mean(torch.stack(head_outs))

class RGAT(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_heads, dropout=0.5, num_layers=2, edge_embed_dim=50):
        super(RGAT, self).__init__()
        fan_in = in_dim
        self.gat_layers = nn.ModuleList()
        for i in range(num_layers - 1):
            gat_layer = MultiHeadGATLayer(fan_in, hidden_dim, num_heads, edge_embed_dim=edge_embed_dim)
            fan_in = hidden_dim * num_heads
            self.gat_layers.append(gat_layer)
        gat_layer = MultiHeadGATLayer(fan_in, out_dim, 1)
        self.gat_layers.append(gat_layer)

        self.dropout = nn.Dropout(dropout)

    def forward(self, g, h):
        for gat_layer in self.gat_layers:
            h = F.elu(self.dropout(gat_layer(g, h)))
        return h
