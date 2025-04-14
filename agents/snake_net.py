import torch
import torch.nn as nn
import torch.nn.functional as F

class SnakeNet(nn.Module):
    """
    A simple feed-forward neural network used as the Snake agent's brain.
    Input: 1326 neurons (flattened vision matrix + food distance and normalized length)
    Hidden Layers: 128 and 64 neurons with ReLU activation.
    Output: 4 neurons corresponding to actions: LEFT, RIGHT, UP, DOWN.
    """
    def __init__(self, input_size=1326, hidden1=128, hidden2=64, output_size=4):
        super(SnakeNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.out = nn.Linear(hidden2, output_size)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.out(x)
        x = F.softmax(x, dim=1)  # Output a probability distribution
        return x

if __name__ == '__main__':
    model = SnakeNet()
    print(model)
