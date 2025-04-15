import torch
from agents.snake_net import SnakeNet
from ga.evolution import run_evolution, POPULATION_SIZE, device
from ga.operators import apply_mutation

def create_seed_population(seed_model_path, population_size=POPULATION_SIZE) -> list[SnakeNet]:
    # Load the best pre-trained model.
    seed_model = SnakeNet().to(device)
    seed_model.load_state_dict(torch.load(seed_model_path, map_location=device))
    seed_model.eval()
    
    # Create a seeded population.
    population = []
    # Insert the seed model directly as one elite.
    population.append(seed_model)
    
    # Fill the rest of the population by copying and slightly mutating the seed.
    for _ in range(population_size - 1):
        candidate = SnakeNet().to(device)
        candidate.load_state_dict(seed_model.state_dict())
        # Apply a small mutation for diversity.
        candidate_state = candidate.state_dict()
        mutated_state = apply_mutation(candidate_state, mutation_rate=0.05, sigma=0.05)
        candidate.load_state_dict(mutated_state)
        population.append(candidate)
    return population

def continue_training(seed_model_path, extra_generations=50, render=False):
    print("Resuming training from:", seed_model_path)
    population = create_seed_population(seed_model_path)
    best_model, best_fitness = run_evolution(render=render, initial_population=population, num_generations=extra_generations)
    return best_model, best_fitness

if __name__ == "__main__":
    # Path to the existing best model.
    seed_model_path = "best_model.pth"
    extra_generations = 100  # Or any number of extra generations you wish to run.
    best_model, best_fitness = continue_training(seed_model_path, extra_generations=extra_generations, render=False)
    print("Fine tuning complete. Best Fitness:", best_fitness)
    # Save the fine tuned model.
    torch.save(best_model.state_dict(), "best_model_finetuned.pth")
    print("Fine tuned model saved as best_model_finetuned.pth")
