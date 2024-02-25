import matplotlib.pyplot as plt
import numpy as np
from numpy import linalg as LA
from matplotlib.colors import ListedColormap, BoundaryNorm
import scipy.linalg as spla
from ndsolver.utils.cubic_solver import vectorized_depressed_cubic_roots


class MM:
    """Class for testing marginal minimization using the quartic descent algorithm.
    
    The results of various calculations are stored in the following fields
     - _dissim: (n, n) the dissimilarity matrix
     - _cost: float the qMDS cost of the current embedding
     - _marginal_costs: (n,) the marginal costs of the current embedding
     - _grad: (m, n) the gradient of the current embedding
    """
    
    def __init__(self, X: np.ndarray, labels: list, m=2, init: np.ndarray = None, init_scaling_factor=0.1):
        """Initializes a MM object for performing marginal minimiztion with the specified parameters.

        Args:
            X (np.ndarray): (d, n) the input data points
            labels (np.array): (n,) the labels for the input data points
            m (int, optional): The embedding dimensionality. Defaults to 2.
            init (np.ndarray, optional): The initial embedding to improve. Random normal distribution if None. Defaults to None.
            init_scaling_factor (float, optional): The scaling factor of the random normal distribution. Defaults to 0.1.
        """
        self.X = X
        self.labels = labels
        self.d, self.n = X.shape
        self.m = m
        self.Y = np.random.randn(m, self.n) * init_scaling_factor if init is None else init
        self.Y -= np.mean(self.Y, axis=1, keepdims=True)
        # Fields used for retrieval of most recently calculated values
        self._dissim = None
        self._cost = None
        self._marginal_costs = None
        self._grad = None


    def dissim(self, Y: np.ndarray) -> np.ndarray:
        """Calculates the dissimilarity matrix between the input data points and a potential embedding.
        The result of the calculation is stored in the _dissim field.

        Args:
            Y (np.ndarray): (m, n) the potential embedding

        Returns:
            np.ndarray: (n, n) the dissimilarity matrix
        """
        X, d, m, n = self.X, self.d, self.m, self.n
        self._dissim = (X.T * X.T) @ np.ones((d, n)) - 2 * X.T @ X + np.ones((n, d)) @ (X * X) - (
                       (Y.T * Y.T) @ np.ones((m, n)) - 2 * Y.T @ Y + np.ones((n, m)) @ (Y * Y)).astype('float64')
        return self._dissim
    
    
    def cost(self, Y: np.ndarray) -> float:
        """Computes the qMDS cost of a potential embedding of the X data.
        The result of the calculation is stored in the _cost field.

        Args:
            Y (np.ndarray): (m, n) the potential embedding

        Returns:
            float: The qMDS cost of the potential embedding
        """
        self.dissim(Y)
        self._cost = np.tensordot(self._dissim, self._dissim) / 2
        self._cost /= self.n**2
        return self._cost


    def marginal_costs(self, Y: np.ndarray) -> np.ndarray:
        """Computes the marginal costs for each element in a potential embedding of the X data.
        The result of the calculation is stored in the _marginal_costs field.

        Args:
            Y (np.ndarray): (m, n) the potential embedding

        Returns:
            np.ndarray: (n,) the marginal costs of the potential embedding
        """
        self.dissim(Y)
        self._marginal_costs = np.sum(self._dissim**2, axis=1).astype('float64') / 2
        self._marginal_costs /= self.n
        return self._marginal_costs
    
    
    def grad(self, Y: np.ndarray, normalize=False, epsilon_normalize=False, epsilon=1e-8) -> np.ndarray:
        """Computes the gradient of the cost function for a potential embedding of the X data.
        The result of the calculation is stored in the _grad field.

        Args:
            Y (np.ndarray): (m, n) the potential embedding
            normalize (bool, optional): Whether to normalize by dividing by the norm. Defaults to False.
            epsilon_normalize (bool, optional): whether to normalize by dividing by the max of the norm + epsilon. Defaults to False.
            epsilon (_type_, optional): a small value used to prevent division by zero in epsilon normalization. Defaults to 1e-8.

        Returns:
            np.ndarray: (m, n) the gradient of the cost function
        """
        self.dissim(Y)
        # Calculate coordinate differences
        coord_diff = Y[:, :, None] - Y[:, None, :]
        # Interaction term using the current dissimilarity matrix
        interaction = coord_diff * self._dissim
        # Gradient calculation
        self._grad = np.sum(interaction, axis=1) / self.n
        # Normalize gradient
        if epsilon_normalize:    
            norm = LA.norm(self._grad, axis=0)
            self._grad = norm / (np.max(norm) + epsilon) 
        elif normalize:
            self._grad /= LA.norm(self._grad, axis=0)
        return self._grad


    def update_descent(self, iterations: int, learning_rate: float, schedule=lambda i: 1, normalize_grad=False, epsilon_normalize=False, epsilon=1e-8):
        for i in range(iterations):
            # Calculate the gradient using the grad method
            self.grad(self.Y, normalize_grad, epsilon_normalize, epsilon)
            # Update Y using the gradient
            self.Y -= learning_rate * schedule(i) * self._grad


    def update_batched(self, batches=1):
        # This method updates the psi and phi matrices and then the Y embedding
        batch_size = self.n // batches
        for _ in range(batches):
            # generate moment matrices based on current X and Y
            [psi, phi] = self.find_moment_matrices(self.X, self.Y)
            # perform desent on the Y embedding
            Y = self.quartic_descent_vectorized(psi, phi.T)
            # calculate the batch indices
            indices = np.random.choice(self.n, batch_size, replace=False)
            # update that batch in the Y embedding
            self.Y[:, indices] = Y.T[:, indices]
            self.Y -= np.mean(self.Y, axis=1, keepdims=True)
            self._cost = self.cost(self.Y)


    def update(self, mc_threshold: float = 0, grad_threshold: float = 0):
        """_summary_

        Args:
            mc_threshold (float, optional): _description_. Defaults to 0.
            grad_threshold (float, optional): _description_. Defaults to 0.
        """
        # This method updates the psi and phi matrices and then the Y embedding
        # generate moment matrices based on current X and Y
        [psi, phi] = self.find_moment_matrices(self.X, self.Y)
        # perform desent on the Y embedding
        Y = self.quartic_descent_vectorized(psi, phi.T)
        # calculate the normalized marginal costs
        marginal_costs = self.marginal_costs(self.Y)
        marginal_min, marginal_max = np.min(marginal_costs), np.max(marginal_costs)
        normalized_costs = (marginal_costs - marginal_min) / (marginal_max - marginal_min)
        # calculate the batch indices
        indices = np.logical_and(
            normalized_costs >= mc_threshold,
            LA.norm(self.grad(self.Y), axis=0) >= grad_threshold
        )
        # update that batch in the Y embedding and recenter Y
        self.Y[:, indices] = Y.T[:, indices]
        self.Y[:, ~indices] = self.Y[:, ~indices] - 0.01 * self._grad[:, ~indices]
        self.Y -= np.mean(self.Y, axis=1, keepdims=True)
        self._cost = self.cost(self.Y)
    
    
    def critical_lengths(self) -> np.ndarray:
        """Calculates the critical lengths of the cubic equation based on 
        the eigenvalues and eigenvectors of the moment matrix psi.

        Returns:
            np.ndarray: roots of the cubic equation
        """
        A, b = self.find_moment_matrices(self.X, self.Y)
        v, V = LA.eig(A)
        k = np.einsum('ikj,ik->ij', V, b.T)
        return vectorized_depressed_cubic_roots(v, k)
    
    
    def eigenvalues(self):
        """Comptues the eigenvalues of the moment matrix psi based on the current X and Y.

        Returns:
            np.ndarray: (n,) the eigenvalues of psi
        """
        A, b = self.find_moment_matrices(self.X, self.Y)
        v, V = LA.eig(A)
        return v
    
    
    # TODO: needs to be refactored
    def quartic_initialization_vectorized(self, A: np.ndarray, b: np.ndarray):
        """
        Creates vectorized initialization for the quartic descent algorithm.

        Args:
            A (np.ndarray): (n, m, m) the vectorized quadratic coefficients of the quartic functions (psi)
            b (np.ndarray): (n, m) the vectorized linear coefficents of the quartic functions (phi)

        Returns:
            np.ndarray: (n, m) matrix of initial values for the quartic descent algorithm.
        """
        # Get broadcasted vector of eigenvalues (n x m) and matrix of eigenvectors (n x m x m)
        v, V = LA.eig(A)

        # TODO: remove this check, or replace it with somehting robust
        if np.any(np.abs(v[:,0]-v[:,1]) < .1):

            k = np.einsum('ikj,ik->ij', V, b)
            # TODO: in the non guassian case, there are likely three roots
            # we think we want the largest root, but we should check this
            lengths = vectorized_depressed_cubic_roots(v,k)[:,0]
            
            mag_k = np.expand_dims(LA.norm(k, axis = 1), axis = 1)
            unit_phi = k/mag_k
            unit_phi *= np.expand_dims(lengths, axis = 1)
            y0 = np.einsum('ijk,ik->ij', V, unit_phi)
            

        else:

            # cap negative eigenvalues at 0
            v = np.maximum(v, 0)  # TODO: undo?
            #  v = np.abs(v)

            # construct k (n x m) by condensing V.T @ b into a matrix
            k = np.einsum('ikj,ik->ij', V, b)
            # if k is zero, it would initialize to (0, 0)
            # k[k == 0] = 1
            for i in range(k.shape[0]):
                for j in range(k.shape[1]):
                    if k[i, j] == 0:
                        k[i, j] = np.random.choice([-1, 1])

            # find indices of deepest well per row (n,)
            wells = v**2 + np.abs(k) * np.sqrt(v)
            indices = np.argmax(wells, axis=1)

            # zero out all indices except the deepest wells
            z0 = np.zeros(b.shape)
            z0[np.arange(b.shape[0]), indices] = 1
            z0 *= np.sqrt(v) * -np.sign(k)

            # calculate y0 by condensing V @ z0 into a matrix
            y0 = np.einsum('ijk,ik->ij', V, z0)

        return y0


    def quartic_descent_vectorized(self, A: np.ndarray, b: np.ndarray, step_size=0.1, iterations=100):
        """Performs a gradient descent on a vectorized quartic function.

        Args:
            A (np.ndarray): (n, m, m) the vectorized quadratic coefficients of the quartic functions (psi)
            b (np.ndarray): (n, m) the vectorized linear coefficents of the quartic functions (phi)
            step_size (float, optional): The relative distance to move by each iteration. Defaults to 0.1.
            iterations (int, optional): The number of iterations to perform. Defaults to 100.

        Returns:
            _type_: _description_
        """
        y = self.quartic_initialization_vectorized(A, b)

        for _ in range(iterations):
            gradient = 4 * (LA.norm(y, axis=1).reshape(b.shape[0], 1)**2 * y - np.einsum('ijk,ik->ij', A, y) + b)
            y -= step_size * gradient / (np.abs(gradient) + step_size**2)

        return y


    def find_moment_matrices(self, X: np.ndarray, Y: np.ndarray):
        """Finds the moment matrices associated with the marginal problem given input points x and current output points y.

        Args:
            X (np.ndarray): (d, n) the input data points
            Y (np.ndarray): (m, n) the embedding points

        Returns:
            tuple: (psi, phi) the moment matrices
        """
        d, n = X.shape
        m = Y.shape[0]

        norm_x_squared = spla.norm(X, axis=0)**2
        norm_y_squared = spla.norm(Y, axis=0)**2

        psi = (norm_x_squared + np.mean(norm_x_squared - norm_y_squared)).reshape(n, 1, 1) * np.eye(m) - 2/n * (Y @ Y.T).reshape(1, m, m)
        phi = 2/n * Y @ X.T @ X + np.mean(Y * (norm_y_squared - norm_x_squared).reshape(n, 1, 1))

        return psi, phi


    def visualize(self):
        current_cost = self.cost(self.Y)
        formatted_cost = f"{current_cost:.4e}"
        unique_labels = np.unique(self.labels)
        colormap = ListedColormap(plt.cm.gist_rainbow(np.linspace(0, 1, len(unique_labels))))
        norm = BoundaryNorm(np.arange(len(unique_labels)+1)-0.5, len(unique_labels))

        plt.figure(figsize=(18, 5))  # Adjusted figure size to accommodate 3 plots

        # First plot: Visualization based on labels
        plt.subplot(1, 3, 1)
        plt.scatter(self.Y[0, :], self.Y[1, :], c=self.labels, cmap=colormap, norm=norm, alpha=0.75)
        cbar = plt.colorbar(ticks=np.arange(len(unique_labels)))
        cbar.set_ticklabels(unique_labels)
        cbar.set_label('Labels')
        plt.title(f'qMDS Cost = {formatted_cost}')

        # Second plot: Visualization based on marginal cost
        plt.subplot(1, 3, 2)
        marginal_costs = self.marginal_costs(self.Y)
        normalized_costs = ((marginal_costs - np.min(marginal_costs)) / (np.max(marginal_costs) - np.min(marginal_costs)))**(2/3)
        plt.scatter(self.Y[0, :], self.Y[1, :], c=normalized_costs, cmap='plasma', alpha=0.95, s=27.5)
        plt.colorbar(label='Normalized Marginal Cost')
        plt.title('Marginal Cost')

        # Third plot: Points with >0.6 normalized cost using original rainbow color scheme and vectors
        plt.subplot(1, 3, 3)
        high_cost_indices = normalized_costs > 0.65  # Find indices of points with high normalized cost
        Y_high_cost = self.Y[:, high_cost_indices]  # Filter Y for points with high normalized cost
        labels_high_cost = self.labels[high_cost_indices]  # Filter labels for points with high normalized cost
        plt.scatter(Y_high_cost[0, :], Y_high_cost[1, :], c=labels_high_cost, cmap=colormap, norm=norm, alpha=0.75)

        cbar = plt.colorbar(ticks=np.arange(len(unique_labels)))
        cbar.set_ticklabels(unique_labels)
        cbar.set_label('Labels')
        plt.title('High Marginal Cost > 0.65')

        plt.tight_layout()
        plt.show()