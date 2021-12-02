from random import randint, random, sample
from operator import add
from functools import reduce
from statistics import mean
import matplotlib.pyplot as plt


class NumberOptimization:

    def __init__(self, target, fitness_func='value', selection_type='roulette', termination_type='value',
                 retain=0, random_select=0, mutate=0, n_bits=8):

        # Rather than use negative numbers directly (and deal with signed bits etc), 
        # just shift everything by half the max range, then account for it at the end
        self.offset = 2**n_bits / 2
        self.target = target + self.offset
        self.n_bits = n_bits

        self.retain = retain
        self.random_select = random_select
        self.mutate = mutate  

        self.fitness_func = fitness_func
        self.selection_type = selection_type    
        self.termination_type = termination_type

    # Utility function to convert integer to a bit string list
    def int_to_bit_list(self, val):
        return list(map(int, format(val, f'0{self.n_bits}b')))
    # Utility function to convert bit string list to integer
    def bit_list_to_int(self, val):
        return int(''.join(map(str, val)), 2)

    # Create an induvidual with a initial random value
    def individual(self):
        return self.int_to_bit_list(randint(0, 2**self.n_bits - 1))

    # Create the population
    def create_population(self, count):
        self.population = [self.individual() for x in range(count)]
        self.fitness_history = [self.grade()]
        self.pop_history = [self.population_mean()]

    # Check fitness by comparing to target value
    def fitness(self, individual):
        if self.fitness_func == 'value':
            # Convert back to decimal value and return difference
            return abs(self.target-self.bit_list_to_int(individual))

        elif self.fitness_func == 'bits':
            # compare bit strings and return number of bit differences
            diff = 0
            target_bit_list = self.int_to_bit_list(int(self.target))
            for bit1, bit2 in zip(individual, target_bit_list):
                if bit1 != bit2:
                    diff += 1
            return diff

        else:
            raise NotImplementedError

    # Average fitness of the population
    def grade(self):
        summed = reduce(add, (self.fitness(x) for x in self.population), 0)
        return summed / (len(self.population) * 1.0)

    # return two parents from population for crossover
    def selection(self):
        parents = []

        if self.selection_type == 'random':            
            for n in range(2):
                parents.append(self.population[randint(0, len(self.population)-1)])

        elif self.selection_type == 'roulette':
            # reciprocal - fitness of 0 = best
            fitness = [1.0/self.fitness(x) if self.fitness(x) !=0 else 1 for x in self.population]
            fitness_sum = sum(fitness)
            # normalize
            fitness = [x/fitness_sum for x in fitness]

            for n in range(2):
                sel = random()
                p = 0
                for x,fit in zip(self.population, fitness):
                    p += fit
                    if p > sel:
                        parents.append(x)
                        break

        elif self.selection_type == 'tournament':
            for n in range(2):
                sampled = sample(list(zip(self.population, [self.fitness(x) for x in self.population])),3)
                winner = min(sampled, key=lambda x: x[1])
                parents.append(winner[0])

        elif self.selection_type == 'rank':
            graded = [ (self.fitness(x), x) for x in self.population]
            ranked = [ x[1] for x in sorted(graded)]

            for n in range(2):
                sel = randint(0, sum(range(0,len(ranked))))
                p = 0
                for x,fit in zip(ranked, range(len(ranked), 0,-1)):
                    p += fit
                    if p > sel:
                        parents.append(x)
                        break

        else:
            raise NotImplementedError

        return parents                


    # Handle selection, crossover, mutation, termination
    def evolve(self):
        graded = [ (self.fitness(x), x) for x in self.population]
        graded = [ x[1] for x in sorted(graded)]

        # elitism
        if self.retain > 0:
            retain_length = int(len(graded) * self.retain)
            parents = graded[:retain_length]
            # randomly add other individuals to
            # promote genetic diversity
            for individual in graded[retain_length:]:
                if self.random_select > random():
                    parents.append(individual)
        else:
            parents = []        

        # crossover parents to create children
        parents_length = len(parents)
        desired_length = len(self.population) - parents_length
        children = []

        while len(children) < desired_length:
            male, female = self.selection()
            half = int(len(male) / 2)
            child = male[:half] + female[half:] 
            children.append(child)     

        # add children to make up rest of the population
        parents.extend(children)

        # mutate some of the population at random
        for individual in parents:
            if self.mutate > random():
                idx = randint(0, len(individual)-1)
                individual[idx] ^= 1 # Flip bit

        # update population for next generation
        self.population = parents

    # Return the mean value of the population
    def population_mean(self):
        return mean([self.bit_list_to_int(x) for x in self.population]) - self.offset

    # Main optimization loop
    def optimize(self, generations):
        for i in range(generations):
            self.evolve()
            fitness = self.grade()            
            self.fitness_history.append(fitness)
            self.pop_history.append(self.population_mean())
            self.iters = len(self.pop_history)-1

            if self.termination_type == 'value':
                # break if mean population value reaches threshold
                if fitness <= 1.0:
                    break

            elif self.termination_type == 'improvement':
                # break if no improvement in population is seen in next 4 generations - remove those non-improvers
                check = [x >= self.fitness_history[-4] for x in self.fitness_history[-3:] if len(self.fitness_history) >= 4]
                if all(check) and len(self.fitness_history) >= 4:
                    self.fitness_history = self.fitness_history[:-3]
                    self.pop_history = self.pop_history[:-3]
                    self.iters -= 3
                    break

            else:
                raise NotImplementedError






if __name__ == '__main__':
    target = 50

    p_count = 40
    
    retain = 0.1
    random_select = 0.01
    mutate = 0

    generations = 40

    # n = NumberOptimization(target, selection_type='roulette', termination_type='value', retain=retain, random_select=random_select, mutate=mutate)
    # n.create_population(p_count)
    # n.optimize(generations)

    # print(n.fitness_history)
    # print(n.pop_history)

    # plt.plot(n.fitness_history)
    # plt.title('Number Optimization')
    # plt.xlabel('Generations')
    # plt.ylabel('Fitness')
    # plt.show()

    iters = []
    for i in range(101):
        n = NumberOptimization(target, selection_type='roulette', termination_type='value', retain=retain, random_select=random_select, mutate=mutate)
        n.create_population(p_count)
        n.optimize(generations)
        if n.iters != generations:
            iters.append(n.iters)
    print(mean(iters))