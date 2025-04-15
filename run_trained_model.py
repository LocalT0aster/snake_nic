import torch
from agents.snake_net import SnakeNet
from game.snake_game import SnakeGameAI

def run_trained_model():
    # Determine the device (CUDA if available, else CPU).
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    # print(f"Running on device: {device}")
    
    # Instantiate the model with FP16 mode if available, and load the saved weights.
    model = SnakeNet(use_fp16=(use_cuda)).to(device)
    model.load_state_dict(torch.load("best_model.pth", map_location=device))
    model.eval()  # Set the model to evaluation mode

    # Create a game instance with rendering enabled.
    game = SnakeGameAI(render=True)
    game.reset()
    
    # Run the game loop.
    while True:
        # Get the current game state.
        # Expecting a tuple: (vision, extra)
        vision, extra = game.get_state()
        # Convert the vision data (shape: 21x21x3) and extra features (shape: 3) to tensors.
        vision_tensor = torch.tensor(vision, dtype=torch.float16).unsqueeze(0).to(device)  # (1, 21, 21, 3)
        extra_tensor = torch.tensor(extra, dtype=torch.float16).unsqueeze(0).to(device)      # (1, 3)
        
        # Compute the model's action using the two input branches.
        with torch.no_grad():
            output = model(vision_tensor, extra_tensor)
        action = torch.argmax(output, dim=1).item()
        
        # Perform one game step.
        reward, game_over, score = game.play_step(action)
        
        # If the game is over, print the score and restart.
        if game_over:
            print("Game over! Score:", score)
            game.reset()

if __name__ == "__main__":
    run_trained_model()
