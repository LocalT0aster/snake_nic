import torch
import torch.nn as nn
import torch.nn.functional as F

class SnakeNet(nn.Module):
    """
    Multi-branch neural network for the Snake agent.
    
    Architecture:
      1. Vision Branch:
         - Expects input of shape (batch, 21, 21, 3). It is permuted to (batch, 3, 21, 21).
         - Conv1: 3 → 8 filters, 3x3 kernel with padding=1, followed by ReLU.
         - Pool: 2x2 max pooling, reducing spatial dimensions (approximately 21→10).
         - Conv2: 8 → 16 filters, 3x3 kernel with padding=1, followed by ReLU.
         - Pool: 2x2 max pooling, reducing the feature map to roughly 5x5.
         - Flatten: Resulting vector has 16x5x5 = 400 elements.
      
      2. Extra Branch:
         - Expects input of shape (batch, 3) (i.e. normalized food distances and snake length).
         - A fully connected layer maps these 3 features to 5 neurons, followed by ReLU.
      
      3. Fusion and Output:
         - Concatenate the vision branch output (400) and the extra branch output (5) → 405 features.
         - A fully connected (fusion) layer with 50 neurons (ReLU).
         - A final output layer maps 50 → 4 neurons.
         - Softmax activation is applied to yield action probabilities.
    
    Args:
        use_fp16 (bool): If True, both model parameters and inputs are cast to FP16.
    """
    def __init__(self, use_fp16: bool = False):
        super(SnakeNet, self).__init__()
        self.use_fp16 = use_fp16
        
        # Vision branch: process 21x21x3 input.
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1)
        # After two poolings: expected feature map is approximately (16, 5, 5) → Flatten to 400.
        
        # Extra branch: process the additional 3 features.
        self.extra_fc = nn.Linear(3, 5)
        
        # Fusion branch: combine features from both branches.
        self.fusion_fc = nn.Linear(400 + 5, 50)
        self.out = nn.Linear(50, 4)
        
        if self.use_fp16:
            self.half()  # Convert parameters to half precision.
    
    def forward(self, vision: torch.Tensor, extra: torch.Tensor) -> torch.Tensor:
        """
        Forward pass that takes separate inputs for the vision data and extra features.

        Parameters:
            vision (Tensor): Expected shape (batch, 21, 21, 3) in NHWC format.
            extra (Tensor): Expected shape (batch, 3).

        Returns:
            output (Tensor): Softmax probabilities over the 4 actions.
        """
        if self.use_fp16:
            vision = vision.half()
            extra = extra.half()

        # Convert vision from NHWC to NCHW.
        vision = vision.permute(0, 3, 1, 2)  # Now shape: (batch, 3, 21, 21)

        # Process vision branch.
        vision = F.relu(self.conv1(vision))      # (batch, 8, 21, 21)
        vision = self.pool(vision)                 # (batch, 8, ~10, ~10)
        vision = F.relu(self.conv2(vision))        # (batch, 16, ~10, ~10)
        vision = self.pool(vision)                 # (batch, 16, ~5, ~5)
        vision = vision.reshape(-1, 16 * 5 * 5)      # Flatten to (batch, 400)

        # Process extra branch.
        extra = F.relu(self.extra_fc(extra))       # (batch, 5)

        # Fusion: Concatenate the two branches.
        fused = torch.cat((vision, extra), dim=1)    # (batch, 405)
        fused = F.relu(self.fusion_fc(fused))        # (batch, 50)
        fused = self.out(fused)                      # (batch, 4)
        output = F.softmax(fused, dim=1)             # Action probabilities.
        return output


if __name__ == '__main__':
    # Quick test on dummy input.
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    
    # Create model with FP16 if CUDA is available.
    model = SnakeNet(use_fp16=use_cuda).to(device)
    
    # Create dummy inputs:
    # Vision: batch size 1, shape (1, 21, 21, 3)
    dummy_vision = torch.randn(1, 21, 21, 3).to(device)
    # Extra: batch size 1, shape (1, 3)
    dummy_extra = torch.randn(1, 3).to(device)
    
    output = model(dummy_vision, dummy_extra)
    print("Output:", output)
