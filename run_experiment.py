from ga.evolution import run_evolution
import torch

def main():
    print("Starting evolution...")
    best_model, best_fitness = run_evolution(render=False)
    print("Best Model Fitness:", best_fitness)
    # Save the best model for later use.
    torch.save(best_model.state_dict(), "best_model.pth")
    print("Best model saved as best_model.pth")

if __name__ == "__main__":
    main()
