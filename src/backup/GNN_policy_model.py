import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import argparse
import sys
import numpy as np

"""
sys.argv = ['']
del sys

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=10000, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.005, help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=8, help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.6, help='Dropout rate (1 - keep probability).')
parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
parser.add_argument('--patience', type=int, default=100, help='Patience')

args = parser.parse_args()

"""


class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):
        Wh = torch.mm(h, self.W)  # h.shape: (N, in_features), Wh.shape: (N, out_features)
        a_input = self._prepare_attentional_mechanism_input(Wh)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))

        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, Wh)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        N = Wh.size()[0]  # number of nodes

        Wh_repeated_in_chunks = Wh.repeat_interleave(N, dim=0)
        Wh_repeated_alternating = Wh.repeat(N, 1)

        all_combinations_matrix = torch.cat([Wh_repeated_in_chunks, Wh_repeated_alternating], dim=1)

        return all_combinations_matrix.view(N, N, 2 * self.out_features)


class GAT(nn.Module):
    def __init__(self, nfeat, nhid, dropout, alpha, lr):
        """
        nfeat: the number of features
        nhid: the number of hidden features
        dropout: Dropout rate (1 - keep probability)
        alpha: Alpha for the leaky_relu
        """
        super(GAT, self).__init__()
        self.dropout = dropout
        self.encoder = GraphAttentionLayer(nfeat, nhid, dropout, alpha)
        self.fc = nn.Linear(2 * nhid, 1)

        # DEFINE THE OPTIMIZER
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    def forward(self, x, adj):
        """
        input:
        x: [N, d] matrix
            feature matrix, N is the number of nodes, d is the number of features
        adj: [N, N] matrix
            adjacent matrix of the graph
        """
        N = adj.size()[0]  # the number of nodes
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.encoder(x, adj)
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat((x, x.sum(0).unsqueeze(0).repeat(N, 1)), dim=1)
        x = self.fc(x)  # the score of each node

        # return torch.softmax(x, 0)
        return x

    def compute_prob(self, norm_x, norm_adj):
        # change the array into tensor
        norm_x = torch.from_numpy(norm_x).float()
        norm_adj = torch.from_numpy(norm_adj).float()
        prob = self.forward(norm_x, norm_adj).squeeze()

        return prob.cpu().data.numpy()

    def Train(self, obs, target):
        """
        obs: list
             the list of all observations in roll out
        target: numpy array
            the actual sequence
        """
        num_sample = len(obs)
        N = num_sample + 1

        norm_x, norm_adj = obs[0]
        norm_x = torch.from_numpy(norm_x).float()
        norm_adj = torch.from_numpy(norm_adj).float()
        prob_mat = self.forward(norm_x, norm_adj).view(1, N)

        for i in range(1, num_sample):
            norm_x, norm_adj = obs[i]
            norm_x = torch.from_numpy(norm_x).float()
            norm_adj = torch.from_numpy(norm_adj).float()
            prob = self.forward(norm_x, norm_adj).view(1, N)
            prob_mat = torch.cat((prob_mat, prob), dim=0)

        target = torch.from_numpy(target).type(torch.LongTensor)
        criterion = nn.CrossEntropyLoss()

        loss = criterion(prob_mat, target)

        # BACKWARD PASS
        self.optimizer.zero_grad()
        loss.backward()

        # print(list(self.parameters())[0].grad)

        # UPDATE
        self.optimizer.step()

        return
