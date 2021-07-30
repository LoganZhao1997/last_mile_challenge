# The value function model for baseline

import torch


class ValueFunction(object):

    def __init__(self):

        # TODO DEFINE THE MODEL
        self.model = torch.nn.Sequential(
                    torch.nn.Linear(24, 12),
                    torch.nn.ReLU(),
                )
        # DEFINE THE OPTIMIZER

        # RECORD HYPER-PARAMS

        # TEST

    def forward(self):
        return 0

    def compute_values(self, states):
        """
        compute prob distribution over all actions given state
        """
        return 0

    def train(self, states, targets):

        return 0