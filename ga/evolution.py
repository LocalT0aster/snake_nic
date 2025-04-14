import torch
import numpy as np
from game.snake_game import SnakeGameAI
from agents.snake_net import SnakeNet
from ga.operators import tournament_selection, uniform_crossover, apply_mutation

# Determine device: CUDA if available, else CPU.
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
print(f"Using device: {device}")

# GA configuration constants.
POPULATION_SIZE = 50
ELITE_PERCENT = 0.10   # Top 10% are preserved
NUM_GENERATIONS = 100
STEPS_PER_GAME = 500   # Maximum number of steps per simulation

def evaluate_individual(model, render=False):
    """
    Evaluate a given SnakeNet model by running a game simulation.
    
    Parameters:
        model (SnakeNet): The neural network controlling the snake.
        render (bool): Whether to render the game (slower if True).
        
    Returns:
        fitness (float): The fitness score computed based on reward accumulated.
    """
    game = SnakeGameAI(render=render)
    total_reward = 0
    game.reset()
    steps = 0
    while steps < STEPS_PER_GAME:
        state = game.get_state()
        # Convert state to PyTorch tensor and move to device.
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(state_tensor)
        action = torch.argmax(output, dim=1).item()
        reward, game_over, score = game.play_step(action)
        total_reward += reward
        if game_over:
            break
        steps += 1
    # For simplicity, fitness is taken as the accumulated reward.
    fitness = total_reward
    return fitness

def run_evolution(render=False):
    """
    Run the genetic algorithm over a set number of generations.
    
    Parameters:
        render (bool): Whether to render game simulations.
        
    Returns:
        best_model (SnakeNet): The best model from the evolution process.
        best_fitness (float): The best fitness score achieved.
    """
    # Initialize population with random models and move them to device.
    population = [SnakeNet().to(device) for _ in range(POPULATION_SIZE)]
    
    best_fitness = -float('inf')
    best_model = None
    
    for generation in range(NUM_GENERATIONS):
        fitness_scores = []
        for model in population:
            fitness = evaluate_individual(model, render=render)
            fitness_scores.append(fitness)
        
        best_gen_fitness = max(fitness_scores)
        avg_fitness = sum(fitness_scores) / len(fitness_scores)
        print(f"Generation {generation}: Best Fitness = {best_gen_fitness:.2f}, Average Fitness = {avg_fitness:.2f}")
        
        if best_gen_fitness > best_fitness:
            best_fitness = best_gen_fitness
            best_index = fitness_scores.index(best_gen_fitness)
            best_model = population[best_index]
        
        # Sort population by fitness (highest first).
        sorted_indices = np.argsort(fitness_scores)[::-1]
        sorted_population = [population[i] for i in sorted_indices]
        
        # Elitism: preserve top ELITE_PERCENT individuals.
        num_elites = int(POPULATION_SIZE * ELITE_PERCENT)
        new_population = sorted_population[:num_elites]
        
        # Generate new offspring until the population is refilled.
        while len(new_population) < POPULATION_SIZE:
            parent1 = tournament_selection(population, fitness_scores, tournament_size=5)
            parent2 = tournament_selection(population, fitness_scores, tournament_size=5)
            
            parent1_dict = parent1.state_dict()
            parent2_dict = parent2.state_dict()
            child1_dict, child2_dict = uniform_crossover(parent1_dict, parent2_dict, mix_rate=0.5)
            
            child1_dict = apply_mutation(child1_dict, mutation_rate=0.10, sigma=0.1)
            child2_dict = apply_mutation(child2_dict, mutation_rate=0.10, sigma=0.1)
            
            child1 = SnakeNet().to(device)
            child2 = SnakeNet().to(device)
            child1.load_state_dict(child1_dict)
            child2.load_state_dict(child2_dict)
            
            new_population.append(child1)
            if len(new_population) < POPULATION_SIZE:
                new_population.append(child2)
        
        population = new_population[:POPULATION_SIZE]
    
    return best_model, best_fitness

if __name__ == '__main__':
    best_model, best_fitness = run_evolution(render=False)
    print("Evolution completed. Best Fitness:", best_fitness)
