#Normalzed the cost, restructed so that the moment matricies are attributes so to easily plot. Added functionality to swap between the if statements
# in the notebook. Added the updated critical lengths function to account for several roots. Changed visualize to just show phi and Y. Experimented
# with renormalizing Y at each step.

import matplotlib.pyplot as plt
import numpy as np
from numpy import linalg as LA
from matplotlib.colors import ListedColormap, BoundaryNorm
import scipy.linalg as spla
from ndsolver.utils.cubic_solver import vectorized_depressed_cubic_roots, multi_cubic

class MM:
    """Class for testing marginal minimization using the quartic descent algorithm.
    """
    
    def __init__(self, X: np.ndarray, labels: list, m=2, init: np.ndarray = None, init_scaling_factor=1.25):
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
        X -= np.mean(X,axis=1, keepdims=True)
        self.X = X
        
        # center the data in Y; otherwise nothing is valid
        Y = np.random.randn(m, self.n) * init_scaling_factor if init is None else init
        self.Y = Y - np.mean(Y, axis=1, keepdims=True)
        #self. Y /= np.std(self.Y,axis=1, keepdims = True)
        
        self.psi, self.phi = self.find_moment_matrices()
        self.cost = self.get_cost()
        
    
    def MP(Y, psi, phi):
        """Computes the marginal problem values for a given embedding

        Args:
            Y (np.ndarray): (m, n) embedding matrix
            psi (np.ndarray): (n, m, m) tensor of quadratic coefficients
            phi (np.ndarray): (n, m) matrix of linear coefficients

        Returns:
            np.ndarray: (n,) vector of marginal problem values
        """
        return LA.norm(Y, axis=0)**4 - 2 * np.einsum('ij,ijk,ki->i', Y.T, psi, Y) - 4 * np.diagonal(phi @ Y)
    
    
    def get_cost(self,Y=None) -> float:
        if Y is None:
            Y=self.Y
        dissim = (self.X.T * self.X.T) @ np.ones((self.d, self.n)) - 2 * self.X.T @ self.X + np.ones((self.n, self.d)) @ (self.X * self.X) - (
                       (Y.T * Y.T) @ np.ones((self.m, self.n)) - 2 * Y.T @ Y + np.ones((self.n, self.m)) @ (Y * Y)).astype('float64')
        c = ((np.tensordot(dissim, dissim) / 2)/(self.n**2))**(1/4)
        return c
   
   
    def find_moment_matrices(self):
        """Finds the moment matrices associated with the marginal problem at the current state in Y.
        """

        # TODO some parts of the computation below are not changing at all and should not be computed each time -  rather should be stored in init
        norm_x_squared = spla.norm(self.X, axis=0)**2
        norm_y_squared = spla.norm(self.Y, axis=0)**2

        psi = (norm_x_squared + np.mean(norm_x_squared - norm_y_squared)).reshape(self.n, 1, 1) * np.eye(self.m) - 2/(self.n) * (self.Y @ self.Y.T).reshape(1, self.m, self.m)
        phi = 2/(self.n) * self.Y @ self.X.T @ self.X + np.mean(self.Y * (norm_y_squared - norm_x_squared).reshape(self.n, 1, 1))

        return psi, phi
    
    
    def update_state(self, step_size = 0.025, iterations=10, grad = False):
        if grad == True:
            for i in range(iterations//10):
                print(i)
                dissim = (self.X.T * self.X.T) @ np.ones((self.d, self.n)) - 2 * self.X.T @ self.X + np.ones((self.n, self.d)) @ (self.X * self.X) - (
                        (self.Y.T * self.Y.T) @ np.ones((self.m, self.n)) - 2 * self.Y.T @ self.Y + np.ones((self.n, self.m)) @ (self.Y * self.Y)).astype('float64')
                coord_diff = self.Y[:, :, None] - self.Y[:, None, :]
                interaction = coord_diff * dissim
                g = np.sum(interaction, axis=1) / self.n
                self.Y-=step_size*g
            self.Y -=np.mean(self.Y, axis=1, keepdims=True)
        else:
            Y = self.quartic_descent_vectorized(self.psi, self.phi.T, step_size, iterations).T
            self.Y = Y-np.mean(Y, axis=1, keepdims=True)
            #self. Y /= np.std(self.Y,axis=1, keepdims = True)
            self.psi, self.phi = self.find_moment_matrices() 
            self.cost = self.get_cost()
        
        

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

        isotropic_guys = np.abs(v[:,0]-v[:,1]) < 1


        k = np.einsum('ikj,ik->ij', V, b)
        lengths = vectorized_depressed_cubic_roots(v,k)
            
            
        mag_k = np.expand_dims(LA.norm(k, axis = 1), axis = 1)
        I1 = (np.sum(v, axis=1,keepdims = True) / v.shape[1])*lengths**2+3*mag_k*lengths
        indices_ = np.argmax(I1, axis=1)
        opt_lengths = lengths[np.arange(self.n),indices_]
        unit_phi = k/mag_k
        unit_phi *= np.expand_dims(opt_lengths, axis = 1)
        y0_ = np.einsum('ijk,ik->ij', V, unit_phi)
            

        happy_points = np.all(v<0,axis=1)
        v = np.maximum(v, 0)
        for i in range(k.shape[0]):
            for j in range(k.shape[1]):
                if k[i, j] == 0:
                    k[i, j] = np.random.choice([-1, 1])


        wells = v**2 + 4*np.abs(k) * np.sqrt(v)
        indices = np.argmax(wells, axis=1)
        z0 = np.zeros(b.shape)
        z0[np.arange(b.shape[0]), indices] = 1
        z0 *= np.sqrt(v) * -np.sign(k)


        y0 = np.einsum('ijk,ik->ij', V, z0)
        y0[happy_points] = 0
        y0[isotropic_guys] = y0_[isotropic_guys]

        return y0


    def quartic_descent_vectorized(self, A: np.ndarray, b: np.ndarray, step_size=100, iterations=1000):
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
            gradient = 4 * (LA.norm(y, axis=1).reshape(b.shape[0], 1)**(3) * y - np.einsum('ijk,ik->ij', A, y) + b)
            h = MM.exact_line_search(y.T, gradient.T, A, b)
            y -= h * gradient
        return y
    

    def visualize(self):
        current_cost = self.cost
        formatted_cost = f"{current_cost:.4}"
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
        
        plt.subplot(1, 3, 2)
        plt.scatter(self.phi[0, :], self.phi[1, :], c=self.labels, cmap=colormap, norm=norm, alpha=0.75)
        cbar = plt.colorbar(ticks=np.arange(len(unique_labels)))
        cbar.set_ticklabels(unique_labels)
        cbar.set_label('Labels')
        plt.title(f'Phi')
        
        Z = self.quartic_initialization_vectorized(self.psi, self.phi.T).T
        init_cost = self.get_cost(Z)
        formatted_init_cost = f"{init_cost:.4}"
        plt.subplot(1, 3, 3)
        plt.scatter(Z[0, :], Z[1, :], c=self.labels, cmap=colormap, norm=norm, alpha=0.75,clip_on = False)
        cbar = plt.colorbar(ticks=np.arange(len(unique_labels)))
        cbar.set_ticklabels(unique_labels)
        cbar.set_label('Labels')
        plt.title(f'Y0 cost={formatted_init_cost}')
        
        plt.tight_layout()
        #plt.show()
        
        
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
  
        
    def exact_line_search(Y: np.ndarray, G: np.ndarray, psi, phi) -> np.ndarray:
        """Performs vectorized exact line search from the embedding and the gradients.

        Args:
            Y (np.ndarray): (m, n) embedding matrix
            G (np.ndarray): (m, n) gradient matrix
            psi (np.ndarray): (n, m, m) tensor of quadratic coefficients
            phi (np.ndarray): (n, m) matrix of linear coefficients

        Returns:
            np.ndarray: n long vector of step sizes
        """
        # print(Y.shape, G.shape, psi.shape, phi.shape)
        
        a = LA.norm(G, axis=0)**4
        b = -3 * LA.norm(G, axis=0)**2 * np.diagonal(Y.T @ G)
        c = LA.norm(Y, axis=0)**2 * LA.norm(G, axis=0)**2 + 2*np.diagonal(Y.T @ G)**2 - np.einsum('ij,ijk,ki->i', G.T, psi, G)
        d = LA.norm(Y, axis=0)**2 * np.diagonal(Y.T @ G) + np.einsum('ij,ijk,ki->i', Y.T, psi, G) + np.diagonal(phi @ G)
        
        h = multi_cubic(a, b, c, d, all_roots=True).T
        
        # should only ever be one real root (?)
        # MPs = np.zeros((Y.shape[1], 3))
        # for i in range(3):
        #     MPs[:, i] = MM.MP(Y - h[:, i].T * G, psi, phi)
        
        return np.expand_dims(h[:, 0].real, axis=1)