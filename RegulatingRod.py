import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt


class RodClass(object):
    def __init__(self):
        self.dt = 0.1
        self.tau = 0
        self.nbr_of_steps = 10
        self.y0 = [0.1, 0.0]

    def pend(self, y, t):
        theta, omega = y
        dydt = [omega, np.sin(theta) + self.tau]
        return dydt

    def iterate(self):
        t = np.linspace(0, self.dt, self.nbr_of_steps)
        sol = odeint(self.pend, self.y0, t)
        return sol


class ControlSystem(object):
    def __init__(self):
        self.p = 1.1
        self.gamma = 0.15

    def calculate_tourque(self, theta, omega):
        return -self.p * np.sin(theta) - self.gamma * omega


def simulate_reg_system():
    rod = RodClass()
    ctrl_system = ControlSystem()
    nbr_of_time_steps = 1000
    theta_over_time = np.zeros(nbr_of_time_steps)

    for i in range(nbr_of_time_steps):
        theta = rod.y0[0]
        omega = rod.y0[1]
        tau = ctrl_system.calculate_tourque(theta, omega)
        rod.tau = tau

        sol = rod.iterate()
        theta_over_time[i] = sol[-1, 0]
        rod.y0 = sol[-1, :]

    plt.plot(theta_over_time, 'b', label=r'$\theta$')
    plt.show()


class NeuralNetworkRegulation(object):
    def __init__(self, list_of_hidden_neurons):
        # 2 inputs
        # one output
        self.list_of_hidden_neurons = list_of_hidden_neurons
        self.sigmoid_contant = 2
        self.list_of_weights = []
        self.list_of_bias = []

    def update_weights(self, array_of_weights):
        start_index = 0
        self.list_of_weights = []
        self.list_of_bias = []
        previous_hidden = 2
        for i in range(len(self.list_of_hidden_neurons)):
            hidden_neurons = self.list_of_hidden_neurons[i]
            end_index = start_index + hidden_neurons*previous_hidden
            W_arr = array_of_weights[start_index:end_index]
            self.list_of_weights.append(W_arr.reshape((hidden_neurons, previous_hidden)))
            start_index = end_index
            previous_hidden = hidden_neurons

        for i in range(len(self.list_of_hidden_neurons)):
            hidden_neurons = self.list_of_hidden_neurons[i]
            end_index = start_index + hidden_neurons
            b_arr = array_of_weights[start_index:end_index]
            self.list_of_bias.append(b_arr.reshape((hidden_neurons, 1)))

    def sigmoid(self, x):
        exp_eval = np.exp(self.sigmoid_contant * x)
        return exp_eval/(1+exp_eval)

    def calculate_tourque(self, theta, omega):
        input_state = np.array([theta, omega]).reshape((2, 1))
        for i in range(len(self.list_of_weights)):
            W = self.list_of_weights[0]
            b = self.list_of_bias[0]
            linear_out = np.dot(W, input_state) + b
            output = self.sigmoid(linear_out)
            input_state = output

        return output[0, 0]

    def simulate_regulation(self):
        rod = RodClass()
        nbr_of_time_steps = 1000
        theta_over_time = np.zeros(nbr_of_time_steps)

        for i in range(nbr_of_time_steps):
            theta = rod.y0[0]
            omega = rod.y0[1]
            tau = self.calculate_tourque(theta, omega)
            rod.tau = tau

            sol = rod.iterate()
            theta = sol[-1, 0]
            theta_over_time[i] = theta
            rod.y0 = sol[-1, :]

            if not (-np.pi/2 < theta < np.pi/2):
                break
        total_theta = np.sum(np.abs(theta_over_time))
        average_theta = total_theta/i
        fitness = i/average_theta

        return fitness



if __name__ == '__main__':
    simulate_reg_system()
