"""
Functional Gradient Descent: A Toy Example.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


# Create toy dataset.
def create_dataset():
    x = np.linspace(-1, 1, 20)
    y = np.exp(-(((x - 0.5)/0.5) ** 2)) + np.exp(-(((x + 0.5)/0.5) ** 2)) + np.random.randn(20)/20
    return x, y


# The kernel we use.
def poly_kernel(x_1, x_2):
    return np.exp(-(((x_1 - x_2)/0.5)**2))


# Create kernel matrix from dataset.
def create_kernel_matrix(x, kernel_f):
    kernel_matrix = np.zeros((x.size, x.size))
    for i, x_i in enumerate(x):
        for j, x_j in enumerate(x):
            kernel_matrix[i][j] = kernel_f(x_i, x_j)
    return kernel_matrix


# Evaluate f(x) = [f(x_1), ..., f(x_n)] with coefficients alpha.
def evaluate(alpha, kernel_matrix):
    return np.matmul(kernel_matrix, alpha)


# Creates the animation.
def create_animation(x, y, fxs):
    
    fig = plt.figure()
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    
    # Plot true hypothesis
    ax.plot(x, y, '-o', c='tab:blue', label='True Hypothesis')
    ax.set_ylim(np.min(y) - 1, np.max(y) + 1)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    curr_fx, = ax.plot(x, fxs[0], '-', c='tab:red', label='Learned Hypothesis')
    title_text = 'Functional Gradient Descent \n Iteration %d'
    title = plt.text(0.5, 0.85, title_text, horizontalalignment='center', verticalalignment='center', transform=fig.transFigure, fontsize=14)
    ax.legend(loc='lower center')

    # Init only required for blitting to give a clean slate.
    def init():
        curr_fx.set_ydata(np.ma.array(x, mask=True))
        return curr_fx, title,

    def animate(iteration):
        curr_fx.set_ydata(fxs[iteration])
        title.set_text(title_text % iteration)
        return curr_fx, title,

    ani = FuncAnimation(fig, animate, len(fxs.keys()), init_func=init, interval=25, blit=True, repeat=False)
    ani.save('functional_gradient_descent.gif', writer='imagemagick', fps=60)


if __name__ == "__main__":

    # Global constants.
    eta = 0.01              # Learning rate.
    lambda_reg = 0.1        # Regularization coefficient.
    num_iterations = 1000   # Iterations for gradient descent.
    seed = 0                # Random seed.

    # Seed for reproducibility.
    np.random.seed(seed)

    # Obtain data.
    x, y = create_dataset()
    kernel_matrix = create_kernel_matrix(x, poly_kernel)

    # Store fx (training set evaluations) at each iteration. We will use this to create an animation.
    fxs = {}

    # Initialize and iterate.
    alpha = np.random.randn(x.size)
    for iteration in range(num_iterations):
        
        # Evaluate fx using the current alpha.
        fx = evaluate(alpha, kernel_matrix)
        
        # Save.
        fxs[iteration] = fx

        # Compute loss (just for logging!).
        loss = np.sum(np.square(y - fx)) + lambda_reg * (np.matmul(alpha, fx))
        print('Iteration %d: Loss = %0.3f' % (iteration, loss))

        # Compute gradient and update.
        alpha = 2 * eta * (y - fx) + (1 - 2 * lambda_reg * eta) * alpha


    create_animation(x, y, fxs)

