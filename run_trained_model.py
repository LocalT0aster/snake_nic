import torch
from agents.snake_net import SnakeNet
from game.snake_game import SnakeGameAI

def run_trained_model():
    # Determine the device (CUDA if available, else CPU).
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"Running on device: {device}")
    
    # Instantiate the model and load the saved weights.
    model = SnakeNet().to(device)
    model.load_state_dict(torch.load("best_model.pth", map_location=device))
    model.eval()  # Set the model to evaluation mode

    # Create a game instance with rendering enabled.
    game = SnakeGameAI(render=True)
    game.reset()
    
    # Run the game loop.
    while True:
        # Get the current game state.
        state = game.get_state()
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
        
        # Compute the model's action.
        with torch.no_grad():
            output = model(state_tensor)
        action = torch.argmax(output, dim=1).item()
        
        # Perform one game step.
        reward, game_over, score = game.play_step(action)
        
        # If the game is over, print the score and restart.
        if game_over:
            print("Game over! Score:", score)
            game.reset()

if __name__ == "__main__":
    run_trained_model()
