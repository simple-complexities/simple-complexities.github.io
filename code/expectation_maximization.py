"""
Expectation Maximization for Gaussian Mixture Models
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.stats import multivariate_normal

# Create toy dataset.
def create_dataset(num_samples, num_clusters):
    if num_clusters != 2:
        raise NotImplementedError
    gaussian_means = np.array([
        (0, 0),
        (3, 3),
    ])

    gaussian_covars = np.array([
        np.eye(2),
        np.eye(2),
    ])

    # Which cluster does x come from?
    zs = np.random.random_integers(low=0, high=1, size=num_samples)

    # Sample each observation.
    samples = np.empty((num_samples, 2))
    for index, z in enumerate(zs):
        sample = np.random.multivariate_normal(gaussian_means[z], gaussian_covars[z])
        samples[index] = sample
    
    return samples


# Computes the density at a sample, given a Gaussian distribution.
def density(sample, mean, covar):
    return multivariate_normal.pdf(sample, mean=mean, cov=covar)
 

# Computes the responsibilities (gamma(z_nk)) for each n and each k.
def compute_responsibilities(samples, theta):
    mixing_coeffs, gaussian_means, gaussian_covars = theta

    num_samples  = samples.shape[0]
    num_clusters = mixing_coeffs.shape[0]

    # Iterate over each sample and each cluster.
    responsibilities = np.empty((num_samples, num_clusters))
    for n, sample in enumerate(samples):

        # How much does this cluster represent the current point?
        for k in range(num_clusters):
            responsibilities[n][k] = mixing_coeffs[k] * density(sample, gaussian_means[k], gaussian_covars[k])
        
        # Normalize.
        responsibilities[n] /= np.sum(responsibilities[n])

    return responsibilities


# Find new values for parameters.
def update(samples, responsibilities):
    num_samples  = samples.shape[0] 
    num_clusters = responsibilities.shape[1]

    mixing_coeffs   = np.empty(num_clusters)
    gaussian_means  = np.empty((num_clusters, 2))
    gaussian_covars = np.empty((num_clusters, 2, 2))

    num_samples_per_cluster = np.sum(responsibilities, axis=0)
    mixing_coeffs = num_samples_per_cluster / num_samples

    for k in range(num_clusters):
        gaussian_means[k]  = np.average(samples, weights=responsibilities[:, k], axis=0)
        covars = np.empty((num_samples, 2, 2))
        diffs = np.expand_dims(samples - gaussian_means[k], axis=2)
        diffs_T = np.expand_dims(samples - gaussian_means[k], axis=1)
        covars = np.matmul(diffs, diffs_T)
        gaussian_covars[k] = np.average(covars, weights=responsibilities[:, k], axis=0)

    return (mixing_coeffs, gaussian_means, gaussian_covars)


# Computes the current log-likelihood.
def compute_log_likelihood(samples, theta):
    log_likelihood = 0
    for n, sample in enumerate(samples):
        sample_density = 0
        for k in range(num_clusters):
            sample_density += mixing_coeffs[k] * density(sample, gaussian_means[k], gaussian_covars[k]) 
        log_likelihood += np.log(sample_density)

    return log_likelihood

# Creates the animation.
def create_animation(samples, log_likelihood_record, responsibilities_record, mixing_coeffs_record, gaussian_means_record, gaussian_covars_record):

    # Create figure.
    fig = plt.figure()
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    
    # Colormap.
    cmap = matplotlib.cm.get_cmap('viridis')

    # Plot ellipses.
    angle = np.linspace(0, 2*np.pi, 100)
    num_clusters = mixing_coeffs_record.shape[1]

    def compute_ellipse_points(cluster_num, iteration):        
        # Find axis to align with.
        eigenvals, eigenvecs = np.linalg.eig(gaussian_covars_record[iteration][cluster_num])
        axis_x = eigenvals[0]
        axis_y = eigenvals[1]

        ellipse_points_x = axis_x*np.cos(angle)
        ellipse_points_y = axis_y*np.sin(angle)

        ellipse_points = np.vstack((ellipse_points_x, ellipse_points_y))

        # Rotate.
        rot_angle = np.arctan(eigenvecs[0][1]/eigenvecs[0][0])
        rot_matrix = [
            [np.cos(rot_angle), -np.sin(rot_angle)],
            [np.sin(rot_angle),  np.cos(rot_angle)],
        ]
        ellipse_points = np.matmul(rot_matrix, ellipse_points)

        # Translate.
        centre_x = gaussian_means_record[iteration][cluster_num][0]
        centre_y = gaussian_means_record[iteration][cluster_num][1]
        ellipse_points[0, :] += centre_x
        ellipse_points[1, :] += centre_y

        return ellipse_points[0], ellipse_points[1]
    
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xlabel('x')
    ax.set_ylabel('y')

    # Update at each iteration.
    num_iterations = responsibilities_record.shape[0]
    def animate(iteration):
        ax.clear()

        # Plot ellipses.
        ellipses = [plt.plot(*compute_ellipse_points(k, iteration), label='Cluster %d' % k) for k in range(num_clusters)]
        ax.legend(title='Covariance Estimate')

        # Plot points.
        points = ax.scatter(samples[:, 0], samples[:, 1], c=cmap(responsibilities_record[iteration, :, 0]))
        ax.set_xlim((-5, 10))
        ax.set_ylim((-5, 10))

        # Update title.
        plt.title('Expectation Maximization \n Iteration %d' % iteration)

    ani = FuncAnimation(fig, animate, num_iterations, interval=100, blit=False, repeat=True)
    ani.save('expectation_maximization.gif', writer='imagemagick', fps=30)

    # Update at each iteration.
    def animate_likelihoods(iteration):
        ax.clear()

        if iteration > 0:
            # Plot log-likelihoods.
            past = 1
            log_likelihood_range = log_likelihood_record[past: iteration + 1]
            ax.plot(np.arange(past, iteration + 1), log_likelihood_range)
            ax.set_ylim((np.min(log_likelihood_range) - 50, np.max(log_likelihood_range) + 50))
            ax.set_xticks(np.arange(past, iteration + 1, 5))

            # Update title.
            plt.title('Expectation Maximization Log-Likelihood \n Iteration %d' % iteration)

    ani = FuncAnimation(fig, animate_likelihoods, num_iterations, interval=100, blit=False, repeat=True)
    ani.save('expectation_maximization_likelihoods.gif', writer='imagemagick', fps=30)
    plt.show()


if __name__ == "__main__":
    
    # Constants.
    num_samples = 500        # Number of points totally.
    num_clusters = 2         # Number of clusters (distributions in the mixture).
    num_iterations = 50      # Number of iterations of EM.
    seed = 0                 # Random seed.

    # Seed for reproducibility.
    np.random.seed(seed)

    # Get dataset.
    samples = create_dataset(num_samples, num_clusters)

    # Ideally, we would randomly initialize, here (or us). But we will start with a mediocre initialization to see convergence.
    mixing_coeffs = np.array([0.5, 0.5])
    gaussian_means = np.array([
        [0, 10],
        [10, 0],
    ])
    gaussian_covars = np.array([
        np.diag(np.random.uniform(size=2)),
        np.diag(np.random.uniform(size=2)),
    ])
    theta = (mixing_coeffs, gaussian_means, gaussian_covars)

    # Record values at each timestep for plotting later.
    responsibilities_record = np.empty((num_iterations, num_samples, num_clusters))
    log_likelihood_record   = np.empty((num_iterations))
    mixing_coeffs_record    = np.empty((num_iterations, num_clusters))
    gaussian_means_record   = np.empty((num_iterations, num_clusters, 2))
    gaussian_covars_record  = np.empty((num_iterations, num_clusters, 2, 2))

    # Incrementally add points to current training.
    for iteration in range(num_iterations):
        
        # E-step.
        responsibilities = compute_responsibilities(samples, theta)

        # M-step.
        theta = update(samples, responsibilities)
        
        # Log current log-likelihood value.
        log_likelihood = compute_log_likelihood(samples, theta)
        print('Iteration %d: Log-likelihood = %0.3f' % (iteration, log_likelihood))

        # Store for plotting later.
        mixing_coeffs, gaussian_means, gaussian_covars = theta
        log_likelihood_record[iteration]   = log_likelihood
        responsibilities_record[iteration] = responsibilities
        mixing_coeffs_record[iteration]    = mixing_coeffs
        gaussian_means_record[iteration]   = gaussian_means
        gaussian_covars_record[iteration]  = gaussian_covars

    # Print learned parameters.
    print()
    print('Learned Parameters:')
    for k in range(num_clusters):
        print('Cluster %d:' % k)
        print('- Mean: %s' % gaussian_means[k])
        print('- Covariance: \n %s' % gaussian_covars[k])
        print('- Mixing Proportion: %s' % mixing_coeffs[k])
        print()

    # Create an animation of the learning process.
    create_animation(samples, log_likelihood_record, responsibilities_record, mixing_coeffs_record, gaussian_means_record, gaussian_covars_record)


