import torch
import random
import copy

def tournament_selection(population, fitness_scores, tournament_size=5):
    """
    Select one individual from the population using tournament selection.
    
    Parameters:
        population (list): List of individuals.
        fitness_scores (list): Fitness scores corresponding to the individuals.
        tournament_size (int): Number of individuals participating in each tournament.
        
    Returns:
        The selected individual.
    """
    selected_indices = random.sample(range(len(population)), tournament_size)
    best_index = None
    best_fitness = -float('inf')
    for idx in selected_indices:
        if fitness_scores[idx] > best_fitness:
            best_fitness = fitness_scores[idx]
            best_index = idx
    return population[best_index]

def uniform_crossover(parent1_dict, parent2_dict, mix_rate=0.5):
    """
    Perform uniform crossover between two parents' state dictionaries.
    
    Parameters:
        parent1_dict (dict): State dictionary (weights) of parent 1.
        parent2_dict (dict): State dictionary (weights) of parent 2.
        mix_rate (float): Probability of swapping each gene.
        
    Returns:
        Tuple of two child state dictionaries.
    """
    child1_dict = {}
    child2_dict = {}
    
    for key in parent1_dict.keys():
        param1 = parent1_dict[key]
        param2 = parent2_dict[key]
        
        # Flatten tensors for element-wise crossover.
        shape = param1.shape
        param1_flat = param1.view(-1)
        param2_flat = param2.view(-1)
        child1_flat = param1_flat.clone()
        child2_flat = param2_flat.clone()
        
        for i in range(len(param1_flat)):
            if random.random() < mix_rate:
                child1_flat[i] = param2_flat[i]
                child2_flat[i] = param1_flat[i]
        
        child1_dict[key] = child1_flat.view(shape)
        child2_dict[key] = child2_flat.view(shape)
    
    return child1_dict, child2_dict

def apply_mutation(state_dict, mutation_rate=0.10, sigma=0.1):
    """
    Apply Gaussian mutation to an individual's weights.
    
    Parameters:
        state_dict (dict): The state dictionary of the individual's neural network.
        mutation_rate (float): The probability of mutating each gene.
        sigma (float): Standard deviation of the Gaussian noise.
        
    Returns:
        A new state dictionary with mutated weights.
    """
    mutated_state_dict = {}
    for key, tensor in state_dict.items():
        mutated_tensor = tensor.clone()
        flat_tensor = mutated_tensor.view(-1)
        for i in range(len(flat_tensor)):
            if random.random() < mutation_rate:
                # Generate a scalar noise value from a normal distribution.
                noise = torch.normal(0.0, sigma, size=())
                # Alternatively, use noise.item() to extract a Python scalar.
                flat_tensor[i] += noise.item()
        mutated_state_dict[key] = flat_tensor.view(tensor.shape)
    return mutated_state_dict
