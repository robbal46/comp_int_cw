import numpy as np
from random import random, randint

from spike_sort_knn import SpikeSortingKNN

# Optimizes the K-Nearest Neighbours classifier parameters using simulated annealing
# Code adapted from lecture notes (Lab 6)
class SimulatedAnnealing():

    # Constructor
    def __init__(self):
        # Initialise variable for SA
        self.demand = 1.0
        self.alpha = 0.1
        self.iters = 100
        self.d = 0.25

    # Main optimization function
    def optimize(self):
        # Create a random initial solution within valid range 
        intitial_solution = [randint(1,20), randint(1,3)]
        solution = self.anneal(intitial_solution)

        # Optimized parameters
        k = int(round(solution[0]))
        p = int(round(solution[1]))

        return k, p

    # Runs main SA loop
    def anneal(self, solution):
        old_cost = self.cost(solution)

        t = 1.0
        t_min = 0.001

        while t > t_min:
            for i in range(self.iters):
                new_solution = self.neighbour(solution)
                new_cost = self.cost(new_solution)
                ap = self.acceptance_probability(old_cost, new_cost, t)
                if ap > random():
                    solution = new_solution
                    old_cost = new_cost

            t = t * self.alpha
        
        return solution


    def acceptance_probability(self, old_cost, new_cost, t):
        a = np.exp((old_cost-new_cost)/t)
        return a

    def cost(self, supply):

        k = int(round(supply[0]))
        p = int(round(supply[1]))

        self.knn = SpikeSortingKNN(k,p, verbose=False)
        self.knn.train_knn()
        accuracy = self.knn.validate_knn()

        cost = self.demand - accuracy
        return cost

    def neighbour(self, solution):
        delta = np.random.random((2,1))
        scale = np.full((2,1), 2*self.d)
        offset = np.full((2,1), 1.0-self.d)

        m = (delta * scale) + offset

        solution[0] = solution[0] * m[0][0]
        #solution[1] = solution[1] * m[1][0]
        solution[1] = randint(1,3)

        # Prevent zero rounding errors
        if solution[0] < 1:
            solution[0] = 1
        if solution[1] < 1:
            solution[1] = 1

        return solution

if __name__ == '__main__':
    sa = SimulatedAnnealing()
    k,p = sa.optimize()

    print(k)
    print(p)






    