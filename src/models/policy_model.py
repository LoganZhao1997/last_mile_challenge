# The policy model for actor

import torch


class Policy(object):

    def __init__(self):

        # DEFINE THE MODEL
        self.model = torch.nn.Sequential(
                    torch.nn.Linear(24, 12),
                    torch.nn.ReLU(),
                )
        # DEFINE THE OPTIMIZER

        # RECORD HYPER-PARAMS

        # TEST

    def struct2vector(self, x, adj):
        """
        Input:
        x, N * d tensor
            the normalized feature vectors
        adj, N * N tensor
            the normalized adjacent matrix
        Output:
        embed, N * h tensor
            the graph embedding
        Parameters:
            N: the number of nodes
            d: the number of node features
            h: the number of embedding features
        """
        return 0

    def forward(self, x, adj):
        """
        Input:
        x, N * d numpy array
            the normalized feature vectors
        adj, N * N numpy array
            the normalized adjacent matrix
        Output:
        prob, N tensor
            the prob mass distribution
        Parameters:
            N: the number of nodes
            d: the number of node features
        """
        x = torch.from_numpy(x).float()
        adj = torch.from_numpy(adj).float()
        embed = self.struct2vector(x, adj)
        prob = self.model(embed)
        return prob

    def compute_prob(self, states):
        """
        compute prob distribution over all actions given state
        """
        return 0


    def _to_one_hot(self, y, num_classes):
        """
        convert an integer vector y into one-hot representation
        """
        scatter_dim = len(y.size())
        y_tensor = y.view(*y.size(), -1)
        zeros = torch.zeros(*y.size(), num_classes, dtype=y.dtype)
        return zeros.scatter(scatter_dim, y_tensor, 1)

    def train(self, states, actions, Qs):
        """
        states: numpy array (states), size [numsamples]
        actions: numpy array (actions), size [numsamples]
        Qs: numpy array (Q values), size [numsamples]
        """
        return 0

