import torch
import torch.nn as nn
import torch.nn.functional as F

class SnakeNet(nn.Module):
    """
    Multi-branch neural network for the Snake agent.
    
    The network architecture is as follows:
      1. **Vision Branch:**
         - Reshape the first 1323 input elements into a (3, 21, 21) tensor.
         - **Conv1:** 3 input channels → 8 filters, kernel size 3x3 with padding=1, followed by ReLU.
         - **Pool1:** 2x2 max pool (reduces spatial dimensions roughly from 21x21 to 10x10).
         - **Conv2:** 8 input channels → 16 filters, kernel size 3x3 with padding=1, followed by ReLU.
         - **Pool2:** 2x2 max pool (reduces dimensions to 5x5).
         - **Flatten:** The output is flattened to a 400-element vector (16x5x5).
      
      2. **Extra Branch:**
         - Processes the last 3 extra features (normalized food distance and snake length) via one dense layer:
           - Dense layer from 3 → 5 neurons followed by ReLU.
      
      3. **Fusion:**
         - Concatenate the vision branch output (400 units) with the extra branch output (5 units) to get 405 features.
         - **Fusion FC:** A dense layer maps 405 → 50 neurons with ReLU.
         - **Output Layer:** A final dense layer maps 50 → 4 neurons followed by softmax activation.
    
    Args:
        use_fp16 (bool): If True, the model and inputs are cast to FP16 (only advisable on CUDA devices).
    """
    def __init__(self, use_fp16=False):
        super(SnakeNet, self).__init__()
        self.use_fp16 = use_fp16
        
        # Vision branch for processing the 21x21x3 input.
        # We expect the vision vector as the first 1323 elements.
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1)
        # After two poolings: (21,21) → (approx. 10,10) after pool1 then (approx. 5,5) after pool2.
        # Thus, flatten size = 16 * 5 * 5 = 400.
        
        # Extra branch for processing the additional 3 features.
        self.extra_fc = nn.Linear(3, 5)
        
        # Fusion branch: Concatenate the 400 vision features and 5 extra features (total 405).
        self.fusion_fc = nn.Linear(400 + 5, 50)
        self.out = nn.Linear(50, 4)
        
        if self.use_fp16:
            self.half()  # Convert model parameters to half precision.
    
    def forward(self, x):
        """
        Forward pass:
          - x is a tensor of shape (batch, 1326), where:
            - First 1323 elements represent the vision data (21x21x3).
            - Last 3 elements are the extra features (e.g. normalized food distance and snake length).
        """
        if self.use_fp16:
            x = x.half()
        # Split input into vision and extra features.
        vision = x[:, :1323]   # shape: (batch, 1323)
        extra = x[:, 1323:]    # shape: (batch, 3)
        
        # Reshape vision branch input to (batch, 3, 21, 21).
        vision = vision.view(-1, 3, 21, 21)
        
        # Process vision branch.
        vision = F.relu(self.conv1(vision))   # → (batch, 8, 21, 21)
        vision = self.pool(vision)              # → (batch, 8, 10, 10)
        vision = F.relu(self.conv2(vision))     # → (batch, 16, 10, 10)
        vision = self.pool(vision)              # → (batch, 16, 5, 5)
        vision = vision.view(-1, 16 * 5 * 5)      # Flatten to (batch, 400)
        
        # Process extra branch.
        extra = F.relu(self.extra_fc(extra))    # → (batch, 5)
        
        # Fusion: Concatenate and forward through fusion FC and output layer.
        fused = torch.cat((vision, extra), dim=1)  # (batch, 405)
        fused = F.relu(self.fusion_fc(fused))       # (batch, 50)
        fused = self.out(fused)                     # (batch, 4)
        output = F.softmax(fused, dim=1)
        return output

if __name__ == '__main__':
    # Quick test on dummy input.
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    
    # Create model with FP16 if CUDA is available.
    model = SnakeNet(use_fp16=(use_cuda)).to(device)
    dummy_input = torch.randn(1, 1326).to(device)
    output = model(dummy_input)
    print("Output:", output)
