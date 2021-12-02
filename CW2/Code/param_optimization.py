import numpy as np
import matplotlib.pyplot as plt
from statistics import mean

from number_optimization import NumberOptimization


def test_population_size():
    iter_history = []
    pop = range(5, 200, 5)
    for i in pop:
        n = NumberOptimization(100)
        n.create_population(i)
        n.optimize(100)
        iter_history.append(n.iters)

    plt.plot(pop, iter_history)
    plt.xlabel('Population Size')
    plt.ylabel('Generations')
    plt.title('Population Size vs Number of generations')
    plt.show()

def test_mutation():
    iter_history = []
    mut = np.arange(0, 1, 0.05)
    for i in mut:
        n = NumberOptimization(100, mutate=i, retain=0.2, random_select=0.05)
        n.create_population(40)
        n.optimize(50)
        iter_history.append(n.iters)

    plt.plot(mut*100, iter_history)
    plt.xlabel('Mutation chance (%)')
    plt.ylabel('Generations')
    plt.title('Mutation Chance vs Number of generations')
    plt.show()

def test_retain():
    iter_history = []
    ret = np.arange(0.01, 1, 0.01)
    for i in ret:
        n = NumberOptimization(100, retain=i)
        n.create_population(100)
        n.optimize(100)
        iter_history.append(n.iters)

    plt.plot(ret*100, iter_history)
    plt.xlabel('Retained individuals (%)')
    plt.ylabel('Generations')
    plt.title('Previous individuals retained vs Number of generations')
    plt.show()


def test_selection():
    iter_history = []
    methods = ['random', 'roulette', 'tournament', 'rank']
    for sel in methods:
        n = NumberOptimization(100, selection_type=sel, retain=0.1, random_select=0.05)
        n.create_population(100)
        n.optimize(100)
        iter_history.append(n.iters)
        plt.plot(n.fitness_history, label=sel)

    plt.xlabel('Generations')
    plt.ylabel('Population Fitness')
    plt.title('Comparison of selection methods')
    plt.legend()
    plt.show()

def test_termination():    
    iter_history = []
    methods = ['value', 'improvement']
    for t in methods:
        iters = []
        for i in range(21):
            n = NumberOptimization(100, selection_type='roulette', termination_type=t)
            n.create_population(100)
            n.optimize(100)
            iters.append(n.iters)
            
        mean_iters = mean(iters)
        iter_history.append(mean_iters)
        plt.bar(t, mean_iters, label=t)

    plt.xlabel('Termination Methods')
    plt.ylabel('Average number of iterations')
    plt.title('Comparison of termination methods')
    plt.show()


def test_fitness():
    iter_history = []
    methods = ['value', 'bits']
    for fit in methods:
        iters = []
        for i in range(11):
            n = NumberOptimization(100, fitness_func=fit, termination_type='improvement')
            n.create_population(100)
            n.optimize(100)
            iters.append(n.iters)

        mean_iters = mean(iters)
        iter_history.append(mean_iters)
        plt.bar(fit, mean_iters, label=fit)

    print(iter_history)
    plt.xlabel('Fitness Functions')
    plt.ylabel('Average number of iterations')
    plt.title('Comparison of fitness functions')
    plt.show()




#test_population_size()
test_mutation()
#test_retain()
#test_selection()
#test_termination()
#test_fitness()