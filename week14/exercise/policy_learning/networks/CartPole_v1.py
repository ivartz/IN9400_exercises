import torch.nn as nn
import torch.nn.functional as F

class CartPole_v1(nn.Module):
    def __init__(self, size_of_state_space, size_of_action_space):
        super(CartPole_v1, self).__init__()
        self.size_of_state_space  = size_of_state_space
        self.size_of_action_space = size_of_action_space

        self.affine1 = nn.Linear(self.size_of_state_space, 128)
        self.dropout = nn.Dropout(p=0.6)
        self.affine2 = nn.Linear(128, self.size_of_action_space)

    def forward(self, x):
        x = self.affine1(x)
        x = self.dropout(x)
        x = F.relu(x)
        action_scores = self.affine2(x)
        return F.softmax(action_scores, dim=1)