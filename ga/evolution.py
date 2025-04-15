import torch
import numpy as np
import concurrent.futures
import os
from game.snake_game import SnakeGameAI
from agents.snake_net import SnakeNet
from ga.operators import tournament_selection, uniform_crossover, apply_mutation

# Ensure using spawn for multiprocessing with CUDA.
if os.name != "nt":
    import multiprocessing as mp
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

# Determine device.
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
# print(f"Using device: {device}")

# GA configuration constants.
DEFAULT_POPULATION_SIZE = 80
DEFAULT_ELITE_PERCENT = 0.10   # Top 10% are preserved
DEFAULT_NUM_GENERATIONS = 80
STEPS_PER_GAME = 1000           # Maximum number of steps per simulation

def evaluate_individual(model: SnakeNet, render: bool = False) -> float:
    """
    Evaluate a given SnakeNet model by running a game simulation.
    
    Parameters:
        model (SnakeNet): The neural network controlling the snake.
        render (bool): Whether to render the game (slower if True).
    
    Returns:
        fitness (float): The fitness score computed based on total reward.
    """
    game = SnakeGameAI(render=render)
    total_reward = 0.0
    game.reset()
    steps = 0
    while steps < STEPS_PER_GAME:
        # Get state as a tuple: (vision, extra)
        vision, extra = game.get_state()
        # Convert to tensors with FP16.
        vision_tensor = torch.tensor(vision, dtype=torch.float16).unsqueeze(0).to(device)  # (1, 21, 21, 3)
        extra_tensor = torch.tensor(extra, dtype=torch.float16).unsqueeze(0).to(device)      # (1, 3)
        with torch.no_grad():
            output = model(vision_tensor, extra_tensor)
        action = torch.argmax(output, dim=1).item()
        reward, game_over, score = game.play_step(action)
        total_reward += reward
        if game_over:
            break
        steps += 1
    return total_reward

def evaluate_individual_worker(state_dict, use_fp16, render):
    """
    Worker function for parallel evaluation. Rebuilds a model from a state dictionary,
    then returns its fitness.
    """
    # Each worker creates its own instance and loads the state.
    model = SnakeNet(use_fp16=use_fp16).to(device)
    model.load_state_dict(state_dict)
    return evaluate_individual(model, render)

def run_evolution(render: bool = False,
                  initial_population: list[SnakeNet] | None = None,
                  num_generations: int = DEFAULT_NUM_GENERATIONS,
                  population_size: int = DEFAULT_POPULATION_SIZE,
                  elite_percent: float = DEFAULT_ELITE_PERCENT,
                  mutation_rate: float = 0.1,
                  sigma: float = 0.1,
                  max_workers: int = 8  # Adjust based on available resources.
                  ) -> tuple[SnakeNet, float]:
    """
    Run the genetic algorithm over a set number of generations using parallel fitness evaluation.
    
    Parameters:
        render (bool): Whether to render game simulations.
        initial_population (list[SnakeNet] | None): If provided, this population is used as the starting point.
        num_generations (int): Number of generations to run.
        population_size (int): Number of individuals in the population.
        elite_percent (float): Proportion of survivors (elites) carried over.
        mutation_rate (float): Mutation probability per gene.
        sigma (float): Standard deviation for Gaussian mutation.
        max_workers (int): Maximum number of parallel worker processes.
    
    Returns:
        tuple[SnakeNet, float]: The best model and its fitness.
    """
    if int(population_size * elite_percent) < 1:
        raise ValueError("0 or less survivors after elitism selection.")
    
    # Create new population if none is given.
    if initial_population is None:
        population = [SnakeNet(use_fp16=use_cuda).to(device) for _ in range(population_size)]
    else:
        population = initial_population
        while len(population) < population_size:
            population.append(SnakeNet(use_fp16=use_cuda).to(device))
    
    best_fitness = -float("inf")
    best_model = None

    for generation in range(num_generations):
        # Parallel fitness evaluation.
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Prepare a list of state dictionaries for all models.
            state_dicts = [model.state_dict() for model in population]
            # Launch evaluations in parallel.
            futures = [executor.submit(evaluate_individual_worker, sd, use_cuda, render) for sd in state_dicts]
            fitness_scores = [f.result() for f in futures]
        
        best_gen_fitness = max(fitness_scores)
        avg_fitness = sum(fitness_scores) / len(fitness_scores)
        
        if best_gen_fitness > best_fitness:
            best_fitness = best_gen_fitness
            best_index = fitness_scores.index(best_gen_fitness)
            best_model = population[best_index]

        # Sort individuals by fitness in descending order.
        sorted_indices = np.argsort(fitness_scores)[::-1]
        sorted_population = [population[i] for i in sorted_indices]

        num_elites = int(population_size * elite_percent)
        new_population = sorted_population[:num_elites]

        elite_fitness_scores = [fitness_scores[i] for i in sorted_indices[:num_elites]]
        avg_elite_fitness = sum(elite_fitness_scores) / num_elites
    
        print(f"Gen {generation}: BestFit= {best_gen_fitness:.2f}, AvgFit= {avg_fitness:.2f}, AvgEliteFit= {avg_elite_fitness:.2f}")

        while len(new_population) < population_size:
            parent1 = tournament_selection(population, fitness_scores, tournament_size=int(population_size * elite_percent))
            parent2 = tournament_selection(population, fitness_scores, tournament_size=int(population_size * elite_percent))

            parent1_dict = parent1.state_dict()
            parent2_dict = parent2.state_dict()
            child1_dict, child2_dict = uniform_crossover(parent1_dict, parent2_dict, mix_rate=0.5)

            child1_dict = apply_mutation(child1_dict, mutation_rate=mutation_rate, sigma=sigma)
            child2_dict = apply_mutation(child2_dict, mutation_rate=mutation_rate, sigma=sigma)

            child1 = SnakeNet(use_fp16=use_cuda).to(device)
            child2 = SnakeNet(use_fp16=use_cuda).to(device)
            child1.load_state_dict(child1_dict)
            child2.load_state_dict(child2_dict)

            new_population.append(child1)
            if len(new_population) < population_size:
                new_population.append(child2)

        population = new_population[:population_size]

    return best_model, best_fitness

if __name__ == "__main__":
    best_model, best_fitness = run_evolution(render=False)
    print("Evolution completed. Best Fitness:", best_fitness)
