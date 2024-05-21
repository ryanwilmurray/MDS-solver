import numpy as np
from numpy import linalg as LA
import math, cmath


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


def parametric_root_finder_1d_old(X: np.ndarray, a, b, c) -> np.ndarray:
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
    q = -a.T @ X - c
    
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
    
    l = r[np.arange(np.shape(p)[0]), np.argmax(Z, axis=1)]
    for i in range(X.shape[1]):
        print(l[i]**3 + p[i] * l[i] + q[i])
    
    return r[np.arange(np.shape(p)[0]), np.argmax(Z, axis=1)]



def multi_cubic(a0, b0, c0, d0, all_roots=True):
    ''' Analytical closed-form solver for multiple cubic equations
    (3rd order polynomial), based on `numpy` functions.

    Parameters
    ----------
    a0, b0, c0, d0: array_like
        Input data are coefficients of the Cubic polynomial::

            a0*x^3 + b0*x^2 + c0*x + d0 = 0

    all_roots: bool, optional
        If set to `True` (default) all three roots are computed and returned.
        If set to `False` only one (real) root is computed and returned.

    Returns
    -------
    roots: ndarray
        Output data is an array of three roots of given polynomials of size
        (3, M) if `all_roots=True`, and an array of one root of size (M,)
        if `all_roots=False`.
    '''

    ''' Reduce the cubic equation to to form:
        x^3 + a*x^2 + bx + c = 0'''
    a, b, c = b0 / a0, c0 / a0, d0 / a0

    # Some repeating constants and variables
    third = 1./3.
    a13 = a*third
    a2 = a13*a13
    sqr3 = math.sqrt(3)

    # Additional intermediate variables
    f = third*b - a2
    g = a13 * (2*a2 - b) + c
    h = 0.25*g*g + f*f*f

    # Masks for different combinations of roots
    m1 = (f == 0) & (g == 0) & (h == 0)     # roots are real and equal
    m2 = (~m1) & (h <= 0)                   # roots are real and distinct
    m3 = (~m1) & (~m2)                      # one real root and two complex

    def cubic_root(x):
        ''' Compute cubic root of a number while maintaining its sign
        '''
        root = np.zeros_like(x)
        positive = (x >= 0)
        negative = ~positive
        root[positive] = x[positive]**third
        root[negative] = -(-x[negative])**third
        return root

    def roots_all_real_equal(c):
        ''' Compute cubic roots if all roots are real and equal
        '''
        r1 = -cubic_root(c)
        if all_roots:
            return r1, r1, r1
        else:
            return r1

    def roots_all_real_distinct(a13, f, g, h):
        ''' Compute cubic roots if all roots are real and distinct
        '''
        j = np.sqrt(-f)
        k = np.arccos(-0.5*g / (j*j*j))
        m = np.cos(third*k)
        r1 = 2*j*m - a13
        if all_roots:
            n = sqr3 * np.sin(third*k)
            r2 = -j * (m + n) - a13
            r3 = -j * (m - n) - a13
            return r1, r2, r3
        else:
            return r1

    def roots_one_real(a13, g, h):
        ''' Compute cubic roots if one root is real and other two are complex
        '''
        sqrt_h = np.sqrt(h)
        S = cubic_root(-0.5*g + sqrt_h)
        U = cubic_root(-0.5*g - sqrt_h)
        S_plus_U = S + U
        r1 = S_plus_U - a13
        if all_roots:
            S_minus_U = S - U
            r2 = -0.5*S_plus_U - a13 + S_minus_U*sqr3*0.5j
            r3 = -0.5*S_plus_U - a13 - S_minus_U*sqr3*0.5j
            return r1, r2, r3
        else:
            return r1

    # Compute roots
    if all_roots:
        roots = np.zeros((3, len(a))).astype(complex)
        roots[:, m1] = roots_all_real_equal(c[m1])
        roots[:, m2] = roots_all_real_distinct(a13[m2], f[m2], g[m2], h[m2])
        roots[:, m3] = roots_one_real(a13[m3], g[m3], h[m3])
    else:
        roots = np.zeros(len(a))  # .astype(complex)
        roots[m1] = roots_all_real_equal(c[m1])
        roots[m2] = roots_all_real_distinct(a13[m2], f[m2], g[m2], h[m2])
        roots[m3] = roots_one_real(a13[m3], g[m3], h[m3])

    return roots
    
    
def parametric_root_finder_1d(X: np.ndarray, a, b, c, w) -> np.ndarray:
    """_summary_

    Args:
        X (np.ndarray): (d, n) data matrix
        a (np.ndarray): (d,) vector of parameters
        b (float): parameter
        c (float): parameter
        w (float): parameter
        
    Returns:
        np.ndarray: (n, 3) array of roots
    """
    
    X_norms_squared = LA.norm(X, axis=0)**2
    M2 = np.mean(X_norms_squared)
    
    A1 = -3 * w
    A2 = -X_norms_squared - M2 + b
    A3 = -a.T @ X - c + X_norms_squared * w
    
    ones = np.ones(X.shape[1])
    r = multi_cubic(ones, A1 * ones, A2, A3, all_roots=True).T
    
    p = (3 * A2 - A1**2) / 3
    q = (2 * A1**3 - 9 * A2 * A3 + 27 * A3) / 27
    Z = r ** 2 * -np.expand_dims(p, axis=1) + 3*r*-np.expand_dims(q, axis=1)
    
    return r[np.arange(np.shape(p)[0]), np.argmin(Z, axis=1)].real


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
    
    a = LA.norm(G, axis=0)**4
    b = -3 * LA.norm(G, axis=0)**2 * np.diagonal(Y.T @ G)
    c = LA.norm(Y, axis=0)**2 * LA.norm(G, axis=0)**2 + 2*np.diagonal(Y.T @ G)**2 - np.einsum('ij,ijk,ki->i', G.T, psi, G)
    d = LA.norm(Y, axis=0)**2 * np.diagonal(Y.T @ G) + np.einsum('ij,ijk,ki->i', Y.T, psi, G) + np.diagonal(phi @ G)
    
    print(a.shape, b.shape, c.shape, d.shape)
    r = multi_cubic(a, b, c, d, all_roots=True).T
    print(r.shape)
