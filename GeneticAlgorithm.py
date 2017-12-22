import numpy as np
import numpy.matlib as matlib


class GeneticAlgorithmBase(object):
    def __init__(self, selection_type, nbr_of_genes):
        # parameters
        self.nbr_of_chromosomes = 100
        self.nbr_of_generations = 100
        self.nbr_of_genes = nbr_of_genes
        self.crossover_prob = 0.7
        self.mutation_prob = 1/self.nbr_of_genes
        self.initial_range = 1
        self.mutation_range = 0.1 * self.initial_range
        self.nbr_of_copies_of_best_individual = 1

        # set selection method
        if selection_type == 'RWS':
            self.perform_selection = self.roulette_wheel_selection
        elif selection_type == 'TS':
            self.perform_selection = self.tournament_select
            self.tournament_para = 0.7
            self.tournament_size = 2
        else:
            raise TypeError('Selection type: ', selection_type, ' does not exist use RWS or TS instead.')

        # storage of best variables
        self.best_chromosome = None
        self.best_chromosome_index = None
        self.best_fitness = 0
        self.best_chromosomes_over_generations = list()
        self.best_fitness_over_generations = list()

        self._fitness_vector = np.zeros(self.nbr_of_chromosomes)
        self._chromosome_matrix = None

    def evaluate_individuals(self):
        """
        use self._chromosome_matrix to evaluate the fitness values in array self._fitness_vector
        :return:
        """
        raise NotImplementedError('Implement the evaluation function')

    def calculate_best_chromosome(self):
        self.best_chromosome_index = np.argmax(self._fitness_vector)
        self.best_chromosome = np.copy(self._chromosome_matrix[self.best_chromosome_index, :])
        self.best_fitness = self._fitness_vector[self.best_chromosome_index]
        self.best_chromosomes_over_generations.append(np.copy(self.best_chromosome))
        self.best_fitness_over_generations.append(self.best_fitness)

    def roulette_wheel_selection(self):
        total_fitness = np.sum(self._fitness_vector)
        probability_array = np.cumsum(total_fitness) / total_fitness
        r = np.random.rand()
        winning_chromosome_index = np.searchsorted(probability_array, r)
        winning_chromosome = self._chromosome_matrix[winning_chromosome_index, :]
        return winning_chromosome

    def tournament_select(self):
        individuals_in_tournament = np.random.choice(self.nbr_of_chromosomes, self.tournament_size, replace=False)
        fitness_of_individuals_in_tournament = self._fitness_vector[individuals_in_tournament]
        sorted_fitness_index_in_tournament = np.argsort(-fitness_of_individuals_in_tournament)

        for i in range(self.tournament_size-1):
            r = np.random.rand()
            if r < self.tournament_para:
                winning_individual_in_tournament = sorted_fitness_index_in_tournament[i]
                index_of_winning_in_population = individuals_in_tournament[winning_individual_in_tournament]
                return self._chromosome_matrix[index_of_winning_in_population, :]

        winning_individual_in_tournament = sorted_fitness_index_in_tournament[-1]
        index_of_winning_in_population = individuals_in_tournament[winning_individual_in_tournament]
        return self._chromosome_matrix[index_of_winning_in_population, :]

    def crossover(self, chromosome1, chromosome2):
        crossover_point = np.random.randint(0, self.nbr_of_genes+1)
        new_chromosome1 = np.concatenate((chromosome1[:crossover_point], chromosome2[crossover_point:]))
        new_chromosome2 = np.concatenate((chromosome2[:crossover_point], chromosome1[crossover_point:]))
        return new_chromosome1, new_chromosome2

    def perform_elitism(self, chromosome_matrix):
        best_chromosomes_repeated = matlib.repmat(self.best_chromosome, self.nbr_of_copies_of_best_individual, 1)
        chromosome_matrix[0:self.nbr_of_copies_of_best_individual, :] = best_chromosomes_repeated
        return chromosome_matrix

    def mutate(self, chromosome_matrix):
        genes_to_mutate_matrix = np.random.random(chromosome_matrix.shape) < self.mutation_prob
        nbr_of_genes_to_mutate = np.sum(genes_to_mutate_matrix)
        mutation_range_vector = self.mutation_range * (2*np.random.rand(nbr_of_genes_to_mutate)-1)
        chromosome_matrix[genes_to_mutate_matrix] += mutation_range_vector
        return chromosome_matrix

    def run(self):
        self._chromosome_matrix = self.initial_range * (2*np.random.rand(self.nbr_of_chromosomes, self.nbr_of_genes)-1)
        for i_generation in range(self.nbr_of_generations):
            self.evaluate_individuals()
            self.calculate_best_chromosome()
            new_chromosome_matrix = np.copy(self._chromosome_matrix)
            for i in range(self.nbr_of_chromosomes//2):
                chromosome1 = self.perform_selection()
                chromosome2 = self.perform_selection()

                r = np.random.rand()
                if r < self.crossover_prob:
                    chromosome1, chromosome2 = self.crossover(chromosome1, chromosome2)

                new_chromosome_matrix[i, :] = chromosome1
                new_chromosome_matrix[i+1, :] = chromosome2

            new_chromosome_matrix = self.mutate(new_chromosome_matrix)

            self._chromosome_matrix = self.perform_elitism(new_chromosome_matrix)


class GeneticAlgorithmTest(GeneticAlgorithmBase):
    def evaluate_individuals(self):
        """
        use self._chromosome_matrix to evaluate the fitness values in array self._fitness_vector
        :return:
        """
        self._fitness_vector = -(self._chromosome_matrix[:, 0] ** 2 + self._chromosome_matrix[:, 1] ** 2)


if __name__ == '__main__':
    g = GeneticAlgorithmTest('TS', 2)
    g.run()
    print(g.best_chromosomes_over_generations)
    print(g.best_fitness_over_generations)
    g2 = GeneticAlgorithmTest('RWS', 2)
    g2.run()
    print(g2.best_fitness)
    print(g2.best_chromosomes_over_generations)


