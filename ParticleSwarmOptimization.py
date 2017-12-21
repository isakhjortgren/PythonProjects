import numpy as np
import numpy.matlib as npml
import matplotlib.pyplot as plt


class PSOBase(object):
    def __init__(self, dimensionality):
        self.positions_range = 2
        self.nbr_of_particles = 30
        self.nbr_of_iterations = 100

        self.maximum_velocity = self.positions_range
        self.c1 = 2
        self.c2 = 2
        self.inertia_weight = 1.4
        self.inertia_weight_lower_bound = 0.3
        self.beta = 0.99

        self.vector_length = dimensionality

        self.positions_matrix = self.positions_range * (2 * np.random.rand(self.nbr_of_particles, self.vector_length)-1)
        self.velocity_matrix = self.maximum_velocity * (2*np.random.rand(self.nbr_of_particles, self.vector_length)-1)
        self.swarm_best_value = -np.inf
        self.swarm_best_position = None
        self.list_of_swarm_best_positions = list()
        self.particle_best_value = np.zeros(self.nbr_of_particles) - np.inf
        self.particle_best_position = np.copy(self.positions_matrix)
        self.list_of_swarm_best_value = list()

    def evaluate_all_particles(self):
        """

        :return: numpy Array of particle values
        """
        raise NotImplementedError('Implement a method that evaluates all particles')

    def update_particle_positions_velocities_and_inertia(self, current_particle_values):
        temp = current_particle_values > self.particle_best_value
        self.particle_best_value[temp] = current_particle_values[temp]
        self.particle_best_position[temp] = self.positions_matrix[temp]

        q = np.random.rand(self.nbr_of_particles, 1)
        cognitive_component = self.c1 * q * (self.particle_best_position - self.positions_matrix)

        best_global_position_matrix = npml.repmat(self.swarm_best_position, self.nbr_of_particles, 1)
        r = np.random.rand(self.nbr_of_particles, 1)
        social_component = self.c2 * r * (best_global_position_matrix - self.positions_matrix)

        new_particle_velocities = self.inertia_weight * self.velocity_matrix + cognitive_component + social_component

        self.positions_matrix = self.positions_matrix + new_particle_velocities
        self.velocity_matrix = new_particle_velocities
        self.velocity_matrix[self.velocity_matrix > self.maximum_velocity] = self.maximum_velocity
        self.velocity_matrix[self.velocity_matrix < -self.maximum_velocity] = -self.maximum_velocity

        if self.inertia_weight > self.inertia_weight_lower_bound:
            self.inertia_weight = self.inertia_weight * self.beta

    def update_swarm_best(self, current_particle_values):
        iteration_best = np.max(current_particle_values)
        index_of_best = np.argmax(current_particle_values)
        # Evaluate a new swarm best value and position
        if iteration_best > self.swarm_best_value:
            self.swarm_best_value = iteration_best
            self.swarm_best_position = self.positions_matrix[index_of_best, :]

        self.list_of_swarm_best_value.append(self.swarm_best_value)
        self.list_of_swarm_best_positions.append(self.swarm_best_position)

    def run_pso(self):
        for i_iteration in range(self.nbr_of_iterations):
            print("Iteration, ", i_iteration + 1, ", out of: ", self.nbr_of_iterations)
            current_particle_values = self.evaluate_all_particles()
            self.update_swarm_best(current_particle_values)
            self.update_particle_positions_velocities_and_inertia(current_particle_values)

    def generate_fitness_plot(self):
        plt.figure()
        plt.plot(self.list_of_swarm_best_value, 'b')
        plt.xlabel('Number of iterations')
        plt.ylabel('Fitness value')
        plt.show()


class PSOFunctionTest(PSOBase):

    def __init__(self):
        super().__init__(2)

    def evaluate_all_particles(self):
        """

        :return: numpy Array of particle values
        """
        fitness = -(self.positions_matrix[:, 0]**2 + self.positions_matrix[:, 1]**2)
        return fitness


class PSOBaseNeuralNetworks(PSOBase):
    def __init__(self, list_of_neuron_in_layers):
        self.list_of_neuron_in_layers = list_of_neuron_in_layers

        # calculate nbr of weights
        vector_length = 0
        for i in range(len(list_of_neuron_in_layers)-1):
            first_neurons = list_of_neuron_in_layers[i]
            second_neuron = list_of_neuron_in_layers[i+1]
            vector_length += first_neurons*second_neuron + second_neuron

        super().__init__(vector_length)
        self.list_of_validation_fitness = list()
        self.validation_fitness = -np.inf

    def get_particle_position_with_best_val_fitness(self):
        index_with_best_val_fitness = np.argmax(self.list_of_validation_fitness)
        return self.list_of_swarm_best_positions[index_with_best_val_fitness]

    def evaluate_validation(self):
        """

        :return: Fitness from the validation
        """
        raise NotImplementedError('Implement a validation method')

    def update_swarm_best(self, current_particle_values):
        iteration_best = np.max(current_particle_values)
        index_of_best = np.argmax(current_particle_values)
        # Evaluate a new swarm best value and position
        if iteration_best > self.swarm_best_value:
            self.swarm_best_value = iteration_best
            self.swarm_best_position = self.positions_matrix[index_of_best, :]
            self.validation_fitness = self.evaluate_validation()

        self.list_of_validation_fitness.append(self.validation_fitness)
        self.list_of_swarm_best_value.append(self.swarm_best_value)
        self.list_of_swarm_best_positions.append(self.swarm_best_position)

    def generate_fitness_plot(self):
        plt.figure()
        plt.plot(self.list_of_swarm_best_value, 'b', label='Training fitness')
        plt.plot(self.validation_fitness, 'r', label='Validation fitness')
        plt.xlabel('Number of iterations')
        plt.legend()
        plt.show()


