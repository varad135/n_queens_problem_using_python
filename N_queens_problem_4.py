#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import random
import matplotlib.pyplot as plt


# In[2]:


N = 8  # For an 8x8 board
POPULATION_SIZE = 100
MAX_GENERATIONS = 1000
MUTATION_RATE = 0.1


# In[3]:


def generate_initial_population(population_size, N):
    return [np.random.permutation(N) for _ in range(population_size)]


# In[4]:


def calculate_fitness(solution):
    attacking_pairs = 0
    for i in range(len(solution)):
        for j in range(i + 1, len(solution)):
            if abs(solution[i] - solution[j]) == j - i:
                attacking_pairs += 1
    # Max pairs - attacking pairs to maximize non-attacking pairs
    return (N*(N-1)/2) - attacking_pairs


# In[5]:


def select_parents(population, fitnesses):
    # Tournament selection
    parents = []
    tournament_size = 5
    for _ in range(2):  # Select 2 parents
        participants = random.sample(list(zip(population, fitnesses)), tournament_size)
        participants.sort(key=lambda x: x[1], reverse=True)
        parents.append(participants[0][0])
    return parents


# In[6]:


def crossover(parent1, parent2):
    # Single point crossover
    point = random.randint(1, N-2)
    child = np.concatenate((parent1[:point], parent2[point:]))
    return child


# In[7]:


def mutate(child):
    if random.random() < MUTATION_RATE:
        i, j = random.sample(range(N), 2)
        child[i], child[j] = child[j], child[i]
    return child


# In[13]:


def plot_chessboard(solution):
    board = np.zeros((N, N))
    board[::2, ::2] = 1
    board[1::2, 1::2] = 1
    
    fig, ax = plt.subplots()
    ax.imshow(board, cmap='binary')
    # Placing the queens on the board
    for i, col in enumerate(solution):
        # Correcting the y-coordinate with N-1-i to flip the board
        ax.text(col, N-1-i, '♛', fontsize=12*3, ha='center', va='center', color='black' if (i + col) % 2 == 0 else 'white')
    plt.axis('off')
    plt.show()


# In[14]:


def genetic_algorithm(N):
    population = generate_initial_population(POPULATION_SIZE, N)
    for generation in range(MAX_GENERATIONS):
        new_population = []
        fitnesses = [calculate_fitness(individual) for individual in population]
        
        # Check for a solution
        if max(fitnesses) == N*(N-1)/2:
            print(f"Solution found at generation {generation}")
            return population[fitnesses.index(max(fitnesses))]
        
        for _ in range(POPULATION_SIZE):
            parent1, parent2 = select_parents(population, fitnesses)
            child = crossover(parent1, parent2)
            child = mutate(child)
            new_population.append(child)
        
        population = new_population
    
    print("Max generations reached without finding a complete solution.")
    return None

# Run the algorithm
solution = genetic_algorithm(N)
if solution is not None:
    print("Solution:", solution)
    plot_chessboard(solution)
else:
    print("No solution was found.")


# In[12]:


print ("thank you")


# In[ ]:




