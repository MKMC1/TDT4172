import numpy as np

class LinearRegression():
    
    def __init__(self, learning_rate = 0.001, epochs = 100):
        # NOTE: Feel free to add any hyperparameters 
        # (with defaults) as you see fit
        self.learning_rate = learning_rate
        self.epochs = epochs
        
    def fit(self, X, y):
        """
        Estimates parameters for the classifier
        
        Args:
            X (array<m,n>): a matrix of floats with
                m rows (#samples) and n columns (#features)
            y (array<m>): a vector of floats
        """
        samples = X.shape[0]
        self.weight = np.random.randn(len(X.shape))
        self.bias = 0
        
        for epoch in range(self.epochs):
            grad_feat =  (-2/samples) * X @ (y - (self.weight*X + self.bias))
            grad_bias = (-2/samples) * np.sum(y - (self.weight*X + self.bias))
            self.weight -= self.learning_rate * grad_feat
            self.bias -= self.learning_rate * grad_bias
            
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
        return self.weight * X + self.bias
    
    def get_params(self):
        return self.weight, self.bias
