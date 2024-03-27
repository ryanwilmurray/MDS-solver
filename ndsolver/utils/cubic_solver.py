import numpy as np
from numpy import linalg as LA
import cmath


def vectorized_depressed_cubic_roots(P: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """Finds all of the cube roots across the system of the equation a^3-P*a-||Q||=0 (post diagonalization)
       
        Args:
            P (np.ndarray): (n, m) the nearly repeated evals of psi
            Q (np.ndarray): (n, m) the vectorized linear coefficents of the quartic functions (phi)

        Returns:
            np.ndarray: (n, 3) array of roots
        """
    # gets the norm of the diagonalized phi
    q = LA.norm(Q, axis=1)
    
    # averages the (already close) eigenvalues; this is assuming m=2
    p = np.sum(P, axis=1) / P.shape[1]
    
    # The discriminant
    D = q**2/4+p**3/27
    
    # Removes need for 'if' statement
    # D[i]>0 -> M[i]=0; D[i]<0 -> M[i]=1
    M = (1-np.sign(D))/2
    
    # same as before
    u1 = -q/2 + np.emath.sqrt(D)
    u2 = -q/2 - np.emath.sqrt(D)
    v1 = np.emath.power(u1, 1/3)
    v2 = np.emath.power(u2, 1/3)
    r1 = (v1+v2).real
    
    # r2 & r3 will be non existent when D>0
    r2 = (-(v1+v2)/2+(v1-v2)*cmath.sqrt(3)/2j).real*M
    r3 = (-(v1+v2)/2-(v1-v2)*cmath.sqrt(3)/2j).real*M
    r = np.column_stack((r1,r2,r3))
    return r



def parametric_root_finder_1d(X: np.ndarray, a, b, c) -> np.ndarray:
    """_summary_

    Args:
        X (np.ndarray): (d, n) data matrix
        a (np.ndarray): (d,) vector of parameters
        b (float): parameter
        c (float): parameter
        
    Returns:
        np.ndarray: (n, 3) array of roots
    """
    # TODO: this step can be cached
    X_norms_squared = LA.norm(X, axis=0)**2
    M2 = np.mean(X_norms_squared)
    
    p = -X_norms_squared - M2 + b
    q = a.T @ X + c
    
    D = q**2/4+p**3/27
    
    # Removes need for 'if' statement
    # D[i]>0 -> M[i]=0; D[i]<0 -> M[i]=1
    M = (1-np.sign(D))/2
    
    # same as before
    u1 = -q/2 + np.emath.sqrt(D)
    u2 = -q/2 - np.emath.sqrt(D)
    v1 = np.emath.power(u1, 1/3)
    v2 = np.emath.power(u2, 1/3)
    r1 = (v1+v2).real
    
    # r2 & r3 will be non existent when D>0
    r2 = (-(v1+v2)/2+(v1-v2)*cmath.sqrt(3)/2j).real*M
    r3 = (-(v1+v2)/2-(v1-v2)*cmath.sqrt(3)/2j).real*M
    r = np.column_stack((r1,r2,r3))
    
    Z = r ** 2 * -np.expand_dims(p, axis=1) + 3*r*-np.expand_dims(q, axis=1)
    
    return r[np.arange(np.shape(p)[0]), np.argmax(Z, axis=1)]
    
    
X = np.random.randn(10, 100)
a = np.random.randn(10)
b = 1
c = 1
r = parametric_root_finder_1d(X, a, b, c)
print(r.shape)