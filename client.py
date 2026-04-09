import flwr as fl
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import log_loss, accuracy_score

class HospitalClient(fl.client.NumPyClient):
    def __init__(self, cid, X_train, y_train):
        self.cid = cid
        self.X_train = X_train
        self.y_train = y_train
        
        # Model for Disease Prediction
        self.model = SGDClassifier(loss='log_loss', penalty='l2', max_iter=1, warm_start=True, learning_rate='constant', eta0=0.01)
        self.model.classes_ = np.array([0, 1])
        # Initialize parameters for the first round (Server will overwrite these)
        self.model.coef_ = np.zeros((1, X_train.shape[1]))
        self.model.intercept_ = np.zeros((1,))

    def get_parameters(self, config):
        return [self.model.coef_, self.model.intercept_]

    def set_parameters(self, parameters):
        self.model.coef_ = parameters[0]
        self.model.intercept_ = parameters[1]

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        
        # Train locally for configured number of epochs
        epochs = config.get("local_epochs", 5)
        for _ in range(epochs):
            self.model.partial_fit(self.X_train, self.y_train, classes=np.array([0, 1]))
        
        return self.get_parameters(config={}), len(self.X_train), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        # Evaluate on the local train set just as a health check
        y_pred = self.model.predict(self.X_train)
        y_prob = self.model.predict_proba(self.X_train)
        loss = log_loss(self.y_train, y_prob, labels=[0, 1])
        acc = accuracy_score(self.y_train, y_pred)
        return float(loss), len(self.X_train), {"accuracy": float(acc)}
