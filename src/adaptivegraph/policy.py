import numpy as np
from typing import Protocol
import random


class BanditPolicy(Protocol):
    def select_action(self, context: np.ndarray) -> int:
        ...
    
    def update(self, context: np.ndarray, action: int, reward: float) -> None:
        ...


class LinUCBPolicy:
    """
    LinUCB with Disjoint Linear Models.
    Each arm (action) has its own ridge regressor (A, b).
    Choice is based on UCB of the estimated reward.
    
    Args:
        n_actions: Number of possible actions.
        feature_dim: Dimension of context vectors.
        alpha: Exploration parameter (higher = more exploration).
        ridge_lambda: Regularization parameter for ridge regression (default 1.0).
    """
    def __init__(self, n_actions: int, feature_dim: int = 32, alpha: float = 1.0, ridge_lambda: float = 1.0):
        self.n_actions = n_actions
        self.feature_dim = feature_dim
        self.alpha = alpha
        self.ridge_lambda = ridge_lambda
        
        # Initialize A as identity matrices scaled by ridge_lambda for each arm
        # Using a 3D array for efficiency: (n_actions, dim, dim)
        self.A = np.array([ridge_lambda * np.eye(feature_dim) for _ in range(n_actions)])
        
        # Initialize b as zero vectors: (n_actions, dim)
        self.b = np.zeros((n_actions, feature_dim))
        
        # Pre-compute inverse? No, do it lazily or use solve.
        # For small d (32), solve is fast.

    def select_action(self, context: np.ndarray) -> int:
        """
        Select action using LinUCB (Upper Confidence Bound) strategy.
        
        Formula: p_a = theta_a^T * x + alpha * sqrt(x^T * A_a^-1 * x)
        where theta_a = A_a^-1 * b_a
        
        Args:
            context: Context vector of shape (feature_dim,).
            
        Returns:
            Index of selected action (0 to n_actions-1).
        """
        # Context shape: (d,)
        # We need to compute p_a for all a
        
        p_values = []
        for a in range(self.n_actions):
            A_a = self.A[a]
            b_a = self.b[a]
            
            # Compute A_a_inv * x and A_a_inv * b_a
            # Ideally use solve
            try:
                # theta_a = np.linalg.solve(A_a, b_a)
                # But we also need x.T * A_a_inv * x.
                # Let v = A_a_inv * x => solve(A_a, x)
                
                A_inv_x = np.linalg.solve(A_a, context)
                theta_a = np.linalg.solve(A_a, b_a)
                
                expected_reward = np.dot(theta_a, context)
                uncertainty = self.alpha * np.sqrt(np.dot(context, A_inv_x))
                
                p = expected_reward + uncertainty
                p_values.append(p)
            except np.linalg.LinAlgError:
                # Fallback if singular (rare with ridge init)
                p_values.append(float('-inf'))

        # Argmax with random tie-breaking
        max_p = np.max(p_values)
        candidates = [i for i, p in enumerate(p_values) if p == max_p]
        return int(random.choice(candidates))

    def update(self, context: np.ndarray, action: int, reward: float) -> None:
        """
        Update policy with observed reward.
        
        Updates: A_a += x * x^T, b_a += r * x
        
        Args:
            context: Context vector of shape (feature_dim,).
            action: Action index that was taken.
            reward: Observed reward (can be positive or negative).
        """
        if action < 0 or action >= self.n_actions:
            return
            
        # Update A += x * x.T
        # Update b += r * x
        
        # Outer product of context
        self.A[action] += np.outer(context, context)
        self.b[action] += reward * context
