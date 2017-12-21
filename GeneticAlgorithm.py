import numpy as np


class GeneticAlgorithmBase(object):
    def __init__(self, selection_type):
        self.nbr_of_chomosomes = 100
        self.nbr_of_iterations = 100
        self.nbr_of_genes = 2
        self.crossover_prob = 0.7
        self.mutation_prob = 1/self.nbr_of_genes

        # set selection method
        if selection_type == 'RWS':
            self.perform_selection = self.roulette_wheel_selection
        elif selection_type == 'TS':
            self.perform_selection = self.tournament_select
        else:
            raise TypeError('Selection type: ', selection_type, ' does not exist use RWS or TS instead.')

    def roulette_wheel_selection(self):
        pass

    def tournament_select(self):
        pass

    def crossover(self, chromosome1, chromosome2):
        pass

    def perform_elitism(self):
        pass

    def mutate(self):
        pass

    def run_GA(self):
        for i in range(self.nbr_of_iterations):
