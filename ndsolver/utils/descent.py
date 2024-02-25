import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Callable, Literal
from abc import ABC, abstractmethod


class Normalization:
    """Contains various static methods for normalizing gradients."""
    
    @staticmethod
    def _handle_zero_division(numerator: np.ndarray, denominator: float) -> np.ndarray:
        """Handles the case where the gradient is zero."""
        return np.zeros_like(numerator) if denominator == 0 else numerator / denominator
    
    @staticmethod
    def none(grad: np.ndarray) -> np.ndarray:
        """No normalization of the gradient."""
        return grad
    
    @staticmethod
    def frobenius(grad: np.ndarray) -> np.ndarray:
        """Frobenius normalization of the gradient."""
        return Normalization._handle_zero_division(grad, np.linalg.norm(grad))
    
    @staticmethod
    def L1(grad: np.ndarray) -> np.ndarray:
        """L1 normalization of the gradient."""
        return Normalization._handle_zero_division(grad, np.linalg.norm(grad, 1))
    
    @staticmethod
    def L2(grad: np.ndarray) -> np.ndarray:
        """L2 normalization of the gradient."""
        return Normalization._handle_zero_division(grad, np.linalg.norm(grad, 2))
    
    @staticmethod
    def max(grad: np.ndarray) -> np.ndarray:
        """Max feature normalization of the gradient."""
        return Normalization._handle_zero_division(grad, np.max(np.abs(grad)))
    
    @staticmethod
    def LInf(grad: np.ndarray) -> np.ndarray:
        """LInf normalization of the gradient."""
        return Normalization._handle_zero_division(grad, np.linalg.norm(grad, np.inf))



class DescentOptimizer(ABC):
    """Abstract class for optimization routines. Do not pass the same instance of this class to multiple parallel gradient descent routines."""

    class State:
        """The state of the optimization routine."""
        
        def __init__(
            self, 
            cost: Callable[[np.ndarray], float], 
            grad: Callable[[np.ndarray], np.ndarray], 
            dims: Tuple[int, int], 
            h: float, 
            max_iterations: int,
            normalizer: Callable[[np.ndarray], np.ndarray]
        ):
            """The state of the gradient descent routine that is relevant to optimization.

            Args:
                cost (Callable[[np.ndarray], float]): the cost function to minimize
                grad (Callable[[np.ndarray], np.ndarray]): the gradient function
                dims (Tuple[int, int]): the dimensions of the input tensor
                h (float): the learning rate
                max_iterations (int): the maximum number of iterations to descend
                normalizer (Callable[[np.ndarray], np.ndarray], optional): the gradient normalization function.
            """
            self.cost = cost
            self.grad = grad
            self.dims = dims
            self.h = h
            self.max_iterations = max_iterations
            self.normalizer = normalizer
        
    @abstractmethod
    def initialize(self, state: State) -> np.ndarray | None:
        """Initializes the state used for the optimization routine.

        Args:
            state (DescentOptimizer.State): the state of the descent routine
            
        Returns:
            np.ndarray | None: the initialization of the input tensor, 
                None lets the gradient descent function initialize it randomly 
        """
    
    @abstractmethod
    def update(self, Y: np.ndarray, state: State) -> Tuple[np.ndarray, bool]:
        """Updates the state of the routine each iteration of descent.

        Args:
            Y (np.ndarray): the current input tensor
            state (DescentOptimizer.State): the state of the descent routine

        Returns:
            Tuple[np.ndarray, bool]: the new input tensor and a boolean indicating whether to stop iterating early
        """


class Momentum(DescentOptimizer):
    """Momentum optimization for gradient descent, which allows past gradients to influence the current step."""
    
    def __init__(self, momentum: float):
        self.momentum = momentum
    
    def initialize(self, state):
        self.velocity = np.zeros(state.dims)
    
    def update(self, Y, state):
        self.velocity = self.momentum * self.velocity - state.h * state.normalizer(state.grad(Y))
        quit = False
        return Y + self.velocity, quit




class DescentArgs:
    """This class that holds the arguments for the gradient descent function."""
    
    def __init__(
        self,
        cost: Callable[[np.ndarray], float],
        grad: Callable[[np.ndarray], np.ndarray],
        dims: Tuple[int, int],
        h: float,
        max_iterations: int = 100,
        init: np.ndarray = None,
        random_init_scaling_factor: float = 1,
        normalizer: Callable[[np.ndarray], np.ndarray] = Normalization.none,
        optimizer: DescentOptimizer = None,
        plot: bool = False
    ):
        """The arguments to the gradient descent function.

        Args:
            cost (Callable[[np.ndarray], float]): the function to minimize
            grad (Callable[[np.ndarray], np.ndarray]): the gradient of the cost function
            dims (Tuple[int, int]): the dimensionality of the input tensor
            h (float): the learning rate of the descent
            max_iterations (int, optional): The maximum number of descent iterations. Defaults to 100.
            init (np.ndarray, optional): The initial value of the input tensor. Defaults to None.
            random_init_scaling_factor (float, optional): The scale of the random normal initialization. Defaults to 1.
            normalizer (Callable[[np.ndarray], np.ndarray], optional): The gradient normalization. Defaults to None.
            optimizer (DescentOptimizer, optional): The gradient descent optimization routine. Defaults to None.
            plot (bool, optional): Whether or not to generate a cost over iterations plot. Defaults to False.
        """
        assert init.shape == dims
        self.cost = cost
        self.grad = grad
        self.dims = dims
        self.h = h
        self.max_iterations = max_iterations
        self.init = init
        self.random_init_scaling_factor = random_init_scaling_factor
        self.normalizer = normalizer
        self.optimizer = optimizer
        self.plot = plot



def gradient_descent(args: DescentArgs) -> np.ndarray:
    """Gradient descent on a given cost function using a given gradient function.

    Args:
        args (DescentArgs): the arguments for the gradient descent function

    Returns:
        np.ndarray: Y, the tensor that minimizes the cost function
    """   
    # start with random embedding
    Y = np.random.randn(*args.dims).astype('float64') * args.random_init_scaling_factor
    
    # initialize the state for the optimization routine
    optimization_state = DescentOptimizer.State(args.cost, args.grad, args.dims, args.h, args.max_iterations, args.normalizer)
    if args.optimizer is not None:
        init = args.optimizer.initialize(optimization_state)
        Y = init if init is not None else Y
    
    # store the cost of the embedding at each iteration
    costs = [float('inf')]
    
    # descend at most max_iterations
    for _ in range(args.max_iterations):
        if args.plot:
            costs.append(args.cost(Y))
        # update the embedding matrix
        if args.optimizer is None:
            Y -= args.h * args.normalizer(args.grad(Y))
        else:
            Y, quit = args.optimizer.update(Y, optimization_state)
            if quit: break

    if args.plot:
        plt.subplot()
        plt.plot(costs[1:])
        plt.title('Embedding Cost vs. Descent Iteration')
        plt.show()
    
    return Y
