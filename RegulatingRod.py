import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt


class RodClass(object):
    def __init__(self):
        self.dt = 0.01
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
        self.gamma = 0.1

    def calculate_tourque(self, theta, omega):
        return -self.p * np.sin(theta) - self.gamma * omega



def simulate_reg_system():
    rod = RodClass()
    ctrl_system = ControlSystem()
    nbr_of_time_steps = 10000
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


if __name__ == '__main__':
    simulate_reg_system()
