import numpy as np

class LogisticRegression():
    
    def __init__(self, learning_rate = 0.01, epochs = 100, tol=1e-9, threshold = 0.5, regularization = 0.1):
        # NOTE: Feel free to add any hyperparameters 
        # (with defaults) as you see fit
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.tol = tol
        self.threshold = threshold
        self.regularization = regularization
        self.weights, self.bias = None, None
        self.losses, self.accuracies = [], []
        
    def fit(self, X, y):
        """
        Estimates parameters for the classifier
        
        Args:
            X (array<m,n>): a matrix of floats with
                m rows (#samples) and n columns (#features)
            y (array<m>): a vector of floats
        """
        if len(X.shape) == 1:
            X = np.reshape(X, (X.shape[0],1))
        
        if len(y.shape) == 1:
            y = np.reshape(y, (y.shape[0],1))
        
        y= y.astype('int64')
        samples, features = X.shape
        self.weights = np.random.randn(features).reshape(features, 1)
        self.bias = 0
        
        for _ in range(self.epochs):
            linear_model = self.linear_model(X)
            y_pred = self.sigmoid_function(linear_model)
            grad_w, grad_b = self.compute_gradients(X, y, y_pred)
            self.update_parameters(grad_w, grad_b)
            
            self.losses.append(self._compute_loss(y, y_pred))
            predictions = np.reshape(np.array(self.predict(X)),(X.shape[0],1))
            acc = self.accuracy(y, predictions)
            self.accuracies.append(acc)
            
    def predict(self, X):
        if len(X.shape) == 1:
            X = np.reshape(X, (X.shape[0],1))
            
        lin_mod = self.linear_model(X)
        y_pred = self.sigmoid_function(lin_mod)
        return [1 if _y > self.threshold else 0 for _y in y_pred]
        
    def linear_model(self, X):
        """
        Generates predictions
        
        Note: should be called after .fit()
        
        Args:
            X (array<m,n>): a matrix of floats with 
                m rows (#samples) and n columns (#features)
            
        Returns:
            A length m array of floats
        """
        return np.dot(X, self.weights) + self.bias

    def sigmoid_function(self, z):
        return 1 / (1 + np.exp(-z))
        
    def _compute_loss(self, y, y_pred):
        y1 = -y *np.log(y_pred + self.tol)
        y2 = -(1 -y)*np.log(1- y_pred + self.tol)
        return np.mean(y1 + y2)
    
    def compute_gradients(self, X, y, y_pred):
        grad_w = np.matmul(X.transpose(), y_pred - y)
        grad_b = y_pred - y
        return grad_w, grad_b
    
    def update_parameters(self, grad_w, grad_b):
        self.weights -= self.learning_rate * grad_w
        self.bias -= self.learning_rate * grad_b
        
    def get_params(self):
        return self.weights, self.bias
    
    def _get_losses(self):
        return self.losses
    
    def _get_accuracies(self):
        return self.accuracies

    def accuracy(self, y, y_pred):
        return np.mean(y == y_pred)