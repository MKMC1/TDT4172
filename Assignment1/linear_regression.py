import numpy as np

class LinearRegression():
    
    def __init__(self, learning_rate = 0.001, epochs = 100):
        # NOTE: Feel free to add any hyperparameters 
        # (with defaults) as you see fit
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights, self.bias = None, None
        self.losses = []
        
    def fit(self, X, y):
        """
        Estimates parameters for the classifier
        
        Args:
            X (array<m,n>): a matrix of floats with
                m rows (#samples) and n columns (#features)
            y (array<m>): a vector of floats
        """
        if len(X.shape) == 1: # Reshape for shape issues
            X = np.reshape(X, (X.shape[0],1))
            y = np.reshape(y, (y.shape[0],1))
            
        self.weights = np.random.randn(X.shape[1]).reshape(X.shape[1], 1)    
        self.bias = 0
        
        for _ in range(self.epochs):
            y_pred = self.predict(X)
            grad_w, grad_b = self.compute_gradients(X, y, y_pred)
            self.update_parameters(grad_w, grad_b)
            
            self.losses.append(self._compute_loss(y, y_pred))
            
    def predict(self, X):
        """
        Generates predictions
        
        Note: should be called after .fit()
        
        Args:
            X (array<m,n>): a matrix of floats with 
                m rows (#samples) and n columns (#features)
            
        Returns:
            A length m array of floats
        """
        if len(X.shape) == 1:
            X = np.reshape(X, (X.shape[0],1))
            
        return np.dot(self.weights, X.T).T + self.bias

    def _compute_loss(self, y, y_pred):
        return np.mean((y - y_pred) ** 2)
    
    def compute_gradients(self, X, y, y_pred):
        """
        Compute gradient for Mean square error for weights and bias separately
        """
        samples = X.shape[0]
        grad_w = (-2/samples) * np.dot(X.T, (y - y_pred))
        grad_b = (-2/samples) * np.sum(y - y_pred)
        return grad_w, grad_b
    
    def update_parameters(self, grad_w, grad_b):
        self.weights -= self.learning_rate * grad_w
        self.bias -= self.learning_rate * grad_b
        
    def get_params(self):
        return self.weights, self.bias
    
    def _get_losses(self):
        return self.losses
