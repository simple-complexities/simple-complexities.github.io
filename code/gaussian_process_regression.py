"""
An animation of Gaussian Process Regression.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Create toy dataset.
def create_dataset(num_samples, error_variance):
    x = np.linspace(-1, 1, num_samples)
    y = np.exp(-(((x - 0.5)/0.5) ** 2)) + np.exp(-(((x + 0.5)/0.5) ** 2)) + np.random.randn(num_samples) * np.sqrt(error_variance)
    return x, y


# Kernel.
def kernel_f(x_1, x_2):
    return np.exp(-(((x_1 - x_2)/0.5)**2))


# Returns the predicted mean and covariance matrix of test samples conditional on the training input.
def predict(kernel_matrix_train, kernel_matrix_train_test, kernel_matrix_test, y, error_variance):
    factor = np.matmul(kernel_matrix_train_test.T, np.linalg.inv(kernel_matrix_train + np.eye(kernel_matrix_train.shape[0])*error_variance))
    mean = np.matmul(factor, y)
    covar = kernel_matrix_test + np.eye(kernel_matrix_test.shape[0])*error_variance - np.matmul(factor, kernel_matrix_train_test)
    return mean, covar


# Creates the animation.
def create_animation(x_train, y_train, x_full, y_full, predictive_means, predictive_vars):
    
    # We need to sort because matplotlib's line markers don't work nicely.
    sorted_indices_full = np.argsort(x_full)
    sorted_indices_train = np.argsort(x_train)
    sorted_indices_test = np.argsort(x_test)

    fig = plt.figure()
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    
    # Plot true hypothesis
    y_masked_train = np.ma.array(y_train, mask=True)
    y_masked_full  = np.ma.array(y_full, mask=True)
    curr_full,  = ax.plot(x_full[sorted_indices_full], y_masked_full, '-', c='tab:red', label='Learned Hypothesis')
    curr_train, = ax.plot(x_train[sorted_indices_train], y_masked_train, 'o', c='tab:blue', label='Training Set Samples')
    ax.plot(x_full[sorted_indices_full], y_full[sorted_indices_full], '-', c='k', label='True Hypothesis')
    
    title_text = 'Gaussian Process Regression \n Iteration %d'
    title = plt.text(0.5, 0.9, title_text, horizontalalignment='center', verticalalignment='center', transform=fig.transFigure, fontsize=14)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_xlim((-1, 1))
    ax.set_ylim((0, 1.5))
    
    def animate(iteration):
        curr_means = predictive_means[iteration][sorted_indices_full]
        curr_vars = predictive_vars[iteration][sorted_indices_full]
        curr_std = np.sqrt(curr_vars)
        curr_full.set_ydata(curr_means)

        y_masked_train.mask[iteration] = False
        curr_train.set_ydata(y_masked_train[sorted_indices_train])

        ax.collections.clear()
        ax.fill_between(x_full[sorted_indices_full], curr_means - curr_std, curr_means + curr_std, color='red', alpha=0.3, label='One-Sigma Error')
        ax.legend(loc='lower center')

        title.set_text(title_text % (iteration + 1))
        return curr_full, curr_train, title,

    ani = FuncAnimation(fig, animate, len(x_train), interval=150, blit=False, repeat=False)
    ani.save('gaussian_process_regression.gif', writer='imagemagick', fps=15)


if __name__ == "__main__":
    
    # Constants.
    num_samples = 100        # Number of points totally.
    error_variance = 1/2000  # Variance of error.
    seed = 0                 # Random seed.

    # Seed for reproducibility.
    np.random.seed(seed)

    # Get dataset.
    x, y = create_dataset(num_samples, error_variance)

    # How many train and test samples?
    num_samples_train = num_samples // 2
    num_samples_test = num_samples - num_samples_train
    
    # Permute, and split.
    permutation = np.random.permutation(num_samples)
    x, y = x[permutation], y[permutation]
    x_train, x_test = x[:num_samples_train], x[num_samples_train:]
    y_train, y_test = y[:num_samples_train], y[num_samples_train:]

    # We will update these one row/column at a time.
    kernel_matrix_train = np.zeros((num_samples_train, num_samples_train))
    kernel_matrix_train_test = np.zeros((num_samples_train, num_samples_test))
    kernel_matrix_train_full = np.zeros((num_samples_train, num_samples))
    predictive_means = np.zeros((num_samples_train, num_samples))
    predictive_vars = np.zeros((num_samples_train, num_samples))

    # Compute kernel matrix for test points, and for all points: these are both fixed!
    kernel_matrix_test = np.zeros((num_samples_test, num_samples_test))
    for i, x_test_i in enumerate(x_test):
        for j, x_test_j in enumerate(x_test[i:], start=i):
            kernel_eval = kernel_f(x_test_i, x_test_j)
            kernel_matrix_test[i, j] = kernel_eval
            kernel_matrix_test[j, i] = kernel_eval

    kernel_matrix_full = np.zeros((num_samples, num_samples))
    for i, x_i in enumerate(x):
        for j, x_j in enumerate(x[i:], start=i):
            kernel_eval = kernel_f(x_i, x_j)
            kernel_matrix_full[i, j] = kernel_eval
            kernel_matrix_full[j, i] = kernel_eval

    # Incrementally add points to current training.
    for t, (x_new, y_new) in enumerate(zip(x_train, y_train)):
        
        # Compute kernel evaluations with previous train samples, and then, with all test samples.
        new_kernel_evals_train = np.array([kernel_f(x_new, x_prev) for x_prev in x_train[:t + 1]])
        new_kernel_evals_test  = np.array([kernel_f(x_new, x_test_sample) for x_test_sample in x_test])
        new_kernel_evals_full  = np.array([kernel_f(x_new, x_sample) for x_sample in x])

        # Fill in a new row (and column) for the new training set sample.
        kernel_matrix_train[t, :t + 1] = new_kernel_evals_train
        kernel_matrix_train[:t + 1, t] = new_kernel_evals_train
        kernel_matrix_train_test[t, :] = new_kernel_evals_test
        kernel_matrix_train_full[t, :] = new_kernel_evals_full

        # Extract out the submatrices we need.
        curr_kernel_matrix_train = kernel_matrix_train[:t + 1, :t + 1]
        curr_kernel_matrix_train_test = kernel_matrix_train_test[:t + 1, :]
        curr_kernel_matrix_train_full = kernel_matrix_train_full[:t + 1, :]
        curr_y_train = y_train[:t + 1]

        # Obtain predictive distribution parameters.
        predictive_mean_test, predictive_covars_test = predict(curr_kernel_matrix_train, curr_kernel_matrix_train_test, kernel_matrix_test, curr_y_train, error_variance)
        predictive_mean, predictive_covars = predict(curr_kernel_matrix_train, curr_kernel_matrix_train_full, kernel_matrix_full, curr_y_train, error_variance)

        # Store these for the plot later.
        predictive_means[t] = predictive_mean
        predictive_vars[t] = np.diag(predictive_covars)

        print('Test L1 Loss: %0.4f' % np.sum(np.abs(predictive_mean_test - y_test)))
    

    create_animation(x_train, y_train, x, y, predictive_means, predictive_vars)


