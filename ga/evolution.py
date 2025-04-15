import torch
import numpy as np
from game.snake_game import SnakeGameAI
from agents.snake_net import SnakeNet
from ga.operators import tournament_selection, uniform_crossover, apply_mutation

# Determine device.
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
print(f"Using device: {device}")

# GA configuration constants.
DEFAULT_POPULATION_SIZE = 50
DEFAULT_ELITE_PERCENT = 0.10   # Top 10% are preserved
DEFAULT_NUM_GENERATIONS = 100
STEPS_PER_GAME = 500   # Maximum number of steps per simulation


def evaluate_individual(model: SnakeNet, render: bool = False) -> float:
    """
    Evaluate a given SnakeNet model by running a game simulation.

    Parameters:
        model (SnakeNet): The neural network controlling the snake.
        render (bool): Whether to render the game (slower if True).

    Returns:
        fitness (float): The fitness score computed based on reward accumulated.
    """
    game = SnakeGameAI(render=render)
    total_reward = 0.0
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
    fitness = total_reward
    return fitness


def run_evolution(render: bool = False,
                  initial_population: list[SnakeNet] | None = None,
                  num_generations: int = DEFAULT_NUM_GENERATIONS,
                  population_size: int = DEFAULT_POPULATION_SIZE,
                  elite_percent: float = DEFAULT_ELITE_PERCENT,
                  mutation_rate: float = 0.1,
                  sigma: float = 0.1
                  ) -> tuple[SnakeNet, float]:
    """
    Run the genetic algorithm over a set number of generations.

    Parameters:
        render (bool): Whether to render game simulations.
        initial_population (list[SnakeNet]): initial population of models. If None, creates a new population of models.
        num_generations (int): The number of generations to run the evolution process.
        population_size (int): The size of the population for the genetic algorithm.
        elite_percent (float): The proportion of the population that is retained as elite models.
        mutation_rate (float): The probability of mutating each gene.
        sigma (float): Standard deviation of the Gaussian noise during mutation.

    Returns:
        result (tuple[SnakeNet, float]): `best_model`, `best_fitness`: The best model from the evolution process; The best fitness score achieved.
    """
    if int(population_size * elite_percent) < 1:
        raise ValueError("0 or less survivors after elitism selection.")

    # If no initial population is provided, create a new one.
    if initial_population is None:
        population = [SnakeNet().to(device) for _ in range(population_size)]
    else:
        population = initial_population
        # If population is smaller than population_size, fill the remainder with new random models.
        while len(population) < population_size:
            population.append(SnakeNet().to(device))

    best_fitness = -float('inf')
    best_model = None

    for generation in range(num_generations):
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

        # Sort population by fitness in descending order.
        sorted_indices = np.argsort(fitness_scores)[::-1]
        sorted_population = [population[i] for i in sorted_indices]

        # Elitism: preserve top elite_percent individuals.
        num_elites = int(population_size * elite_percent)
        new_population = sorted_population[:num_elites]

        # Generate new offspring until the population is refilled.
        while len(new_population) < population_size:
            parent1 = tournament_selection(population, fitness_scores, tournament_size=int(population_size * elite_percent))
            parent2 = tournament_selection(population, fitness_scores, tournament_size=int(population_size * elite_percent))

            parent1_dict = parent1.state_dict()
            parent2_dict = parent2.state_dict()
            child1_dict, child2_dict = uniform_crossover(parent1_dict, parent2_dict, mix_rate=0.5)

            child1_dict = apply_mutation(child1_dict, mutation_rate=mutation_rate, sigma=sigma)
            child2_dict = apply_mutation(child2_dict, mutation_rate=mutation_rate, sigma=sigma)

            child1 = SnakeNet().to(device)
            child2 = SnakeNet().to(device)
            child1.load_state_dict(child1_dict)
            child2.load_state_dict(child2_dict)

            new_population.append(child1)
            if len(new_population) < population_size:
                new_population.append(child2)

        population = new_population[:population_size]

    return best_model, best_fitness


if __name__ == '__main__':
    best_model, best_fitness = run_evolution(render=False)
    print("Evolution completed. Best Fitness:", best_fitness)
