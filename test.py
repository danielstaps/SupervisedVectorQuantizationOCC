import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import cauchy


def studentT(dist, theta):

    prefactor = 1 / (np.pi * theta)

    distribution = 1 / (1 + (dist / (theta**2)))

    probs = prefactor * distribution

    return probs




if __name__ == "__main__":

    distances = np.linspace(start=0, stop=100, num=100)
    theta_boundary = np.linspace(start=0.1, stop=1, num=10)

    plt.figure()
    for tp in theta_boundary:
        #probs = studentT(linspace, tp)
        probs = cauchy.pdf(distances, scale=tp) / tp
        norm = cauchy.pdf(0, scale=tp) / tp
        probs /= norm
    
        plt.plot(distances, probs)

    #plt.show()
