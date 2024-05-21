#Normalzed the cost, restructed so that the moment matricies are attributes so to easily plot. Added functionality to swap between the if statements
# in the notebook. Added the updated critical lengths function to account for several roots. Changed visualize to just show phi and Y. Experimented
# with renormalizing Y at each step.

import matplotlib.pyplot as plt
import numpy as np
from numpy import linalg as LA
from matplotlib.colors import ListedColormap, BoundaryNorm
import scipy.linalg as spla
from ndsolver.utils.cubic_solver import parametric_root_finder_1d
from ndsolver.utils.cubic_solver import vectorized_depressed_cubic_roots
from scipy.spatial.distance import pdist

class MM:
    """Class for testing marginal minimization using the quartic descent algorithm. ONLY WORKS IN 1D FOR NOW.
    """
    
    def __init__(self, X: np.ndarray, labels: list, m=2, init: np.ndarray = None, init_scaling_factor=1.42):
        """Initializes a MM object for performing marginal minimiztion with the specified parameters.

        Args:
            X (np.ndarray): (d, n) the input data points
            labels (np.array): (n,) the labels for the input data points
            m (int, optional): The embedding dimensionality. Defaults to 2.
            init (np.ndarray, optional): The initial embedding to improve. Random normal distribution if None. Defaults to None.
            init_scaling_factor (float, optional): The scaling factor of the random normal distribution. Defaults to 0.1.
        """
        self.m = m
        self.labels = labels
        
        # format the shape of the data X correctly 
        a, b = X.shape
        if a>b:
            X = X.T
        else:
            X = X
        self.d, self.n = X.shape

        # normalize the data; otherwise gradient is completely unstable
        # due to the lines above, these are always the correct axis
        X -= np.mean(X, axis=1, keepdims=True)
        X /= np.std(X, axis=1, keepdims=True)
        self.X = X
        
        # center the data in Y; otherwise nothing is valid
        Y = (np.random.randn(m, self.n) * np.average(pdist(X.T)) * init_scaling_factor) if init is None else init
        self.Y = Y - np.mean(Y, axis=1, keepdims=True)
        
        self.a, self.b, self.c, self.w = self.initialize_parameters()
        # self.Y = np.expand_dims(parametric_root_finder_1d(self.X, self.a, self.b, self.c, self.w), axis=0)
        # self.Y *= np.max(pdist(self.X.T)) / np.average(pdist(self.Y.T)) * 1.42
        # self.a, self.b, self.c, self.w = self.initialize_parameters()
        
        self.cost = self.get_cost()
        
        
    def get_cost(self,Y=None) -> float:
        # if Y is None:
        #     Y=self.Y
        # dissim = (self.X.T * self.X.T) @ np.ones((self.d, self.n)) - 2 * self.X.T @ self.X + np.ones((self.n, self.d)) @ (self.X * self.X) - (
        #                (Y.T * Y.T) @ np.ones((self.m, self.n)) - 2 * Y.T @ Y + np.ones((self.n, self.m)) @ (Y * Y)).astype('float64')
        # c = ((np.tensordot(dissim, dissim) / 2)/(self.n**2))**(1/4)
        # return c
        X, Y = self.X, self.Y.flatten()
        d, n = X.shape
        return 1/8 * np.sum(
            (LA.norm(X[:, i] - X[:, j])**2 - (Y[i] - Y[j])**2)**2
            for j in range(n) for i in range(n)
        )
   
   
    def initialize_parameters(self):
        """Finds the moment matrices associated with the marginal problem at the current state in Y.
        """

        # TODO some parts of the computation below are not changing at all and should not be computed each time -  rather should be stored in init
        norm_x_squared = spla.norm(self.X, axis=0)**2
        norm_y_squared = spla.norm(self.Y, axis=0)**2

        a = 2 / self.n * self.Y @ self.X.T
        b = 3 * np.mean(norm_y_squared)
        c = np.mean(self.Y * (norm_y_squared - norm_x_squared).reshape(self.n, 1, 1))
        w = np.mean(self.Y, axis=1)[0]
        
        return a.flatten(), b, c, w
    
    
    # TODO: this method doesn't seem to account for the -1 in grad_slow
    def grad(self) -> np.ndarray:
        Y = np.expand_dims(parametric_root_finder_1d(self.X, self.a, self.b, self.c), axis=0)
        # Y -= np.mean(Y, axis=1, keepdims=True)
        self.Y = Y
        
        self.dissim(Y)
        # Calculate coordinate differences
        coord_diff = Y[:, :, None] - Y[:, None, :]
        
        X_norms_squared = LA.norm(self.X, axis=0)**2
        M2 = np.mean(X_norms_squared)
        denominator = np.expand_dims(3 * Y**2 - LA.norm(self.X, axis=0)**2 - M2 + self.b, axis=0)
                
        # Interaction term using the current dissimilarity matrix
        interaction = coord_diff * self._dissim / denominator
        
        # Gradient calculation
        self._grad = np.sum(interaction, axis=1) 
        
        params = np.array([*self.a, self.b, self.c])
        tensor = np.einsum('ijk,l->jkl', interaction, params)
        
        self._grape = np.sum(tensor, axis=(0, 1))
        return self._grape / self.n**2
        

    def grad_slow(self):
        Y = np.expand_dims(parametric_root_finder_1d(self.X, self.a, self.b, self.c, self.w), axis=0)
        # Y -= np.mean(Y, axis=1, keepdims=True)
        self.Y = Y
        
        # print("Slow grad:")
        # print("Params: ", self.a, self.b, self.c)
        # print("Y: ", Y)
        
        Y = Y.flatten()
        X_norms_squared = LA.norm(self.X, axis=0)**2
        M2 = np.mean(X_norms_squared)
        
        # print(((LA.norm(self.X[:,0] - self.X[:,0])**2 - (Y[0] - Y[0])**2) * (Y[0] - Y[0])).shape)
        # print(np.array([*self.X[:,0], -Y[0], 1, -LA.norm(self.X[:,0])**2 + 3*Y[0]**2]).shape)
        # print((3 * Y[0]**2 - LA.norm(self.X[:,0])**2 - M2 + self.b - 6 * self.d * Y[0]).shape)
        
        return np.sum(
            ((LA.norm(self.X[:,i] - self.X[:,j])**2 - (Y[i] - Y[j])**2) * (Y[i] - Y[j])) * np.array([*self.X[:,i], -Y[i], 1, -LA.norm(self.X[:,i])**2 + 3*Y[i]**2]) /
            (3 * Y[i]**2 - LA.norm(self.X[:,i])**2 - M2 + self.b - 6 * self.d * Y[i])
            for j in range(self.n) for i in range(self.n)
        ) / self.n**2
        
    def grad_approx(self, h=1e-5):
        def J(X, Y):
            d, n = X.shape
            return 1/8 * np.sum(
                (LA.norm(X[:, i] - X[:, j])**2 - (Y[i] - Y[j])**2)**2
                for j in range(n) for i in range(n)
            )
            
        a, b, c, w, X = self.a, self.b, self.c, self.w, self.X
        d, n = X.shape

        gradient = np.zeros(d + 3)   # d + m + 1
        for i in range(d + 2):
            params = np.array([*a, b, c, w])
            params_plus_h = params.copy()
            params_plus_h[i] += h
            Y = parametric_root_finder_1d(X, params[0:d], params[d], params[d+1], params[d+2])
            Y_plus_h = parametric_root_finder_1d(X, params_plus_h[0:d], params_plus_h[d], params_plus_h[d+1], params_plus_h[d+2])
            gradient[i] = (J(X, Y_plus_h) - J(X, Y)) / h
        return gradient
    
    
    def update_state(self, step_size=1e-3, iterations=100, grad_type=2):
        # get parameters in vector
        params = np.array([*self.a, self.b, self.c, self.w])
        grad = self.grad if grad_type == 1 else self.grad_slow if grad_type == 2 else self.grad_approx
        # gradient descent
        for i in range(iterations):
            params -= step_size * grad()
            self.a, self.b, self.c, self.w = params[:self.d], params[self.d], params[self.d + 1], params[self.d + 2]
        
        self.Y = np.expand_dims(parametric_root_finder_1d(self.X, self.a, self.b, self.c, self.w), axis=0)
        
        
    def critical_lengths(self) -> np.ndarray:
        """Calculates the critical lengths of the cubic equation based on 
        the eigenvalues and eigenvectors of the moment matrix psi.

        Returns:
            np.ndarray: roots of the cubic equation
        """
        #A, b = self.find_moment_matrices(self.X, self.Y)
        v, V = LA.eig(self.psi)
        k = np.einsum('ikj,ik->ij', V, self.phi.T)
        return vectorized_depressed_cubic_roots(v, k)
  
        
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
    
    def visualize(self):
        if self.m == 1 and self.d ==2:
            plt.figure(figsize=(12,5))
            plt.subplot(1, 2, 1)
            colors = self.Y.flatten()
            plt.scatter(*self.X, c=colors, cmap='gist_rainbow')
            plt.colorbar()
            
            plt.subplot(1,2,2)
            plt.hist(colors, bins=100, alpha=0.75, density = True)
            plt.show()