import flwr as fl
import numpy as np
import sys
import os
import json
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import log_loss, accuracy_score

from dataset import generate_synthetic_healthcare_data
from client import HospitalClient
from server import DPFedAvg

def get_evaluate_fn(X_test, y_test):
    """Returns a function for centralized global evaluation."""
    def evaluate(server_round: int, parameters: fl.common.NDArrays, config: dict):
        model = SGDClassifier(loss='log_loss', penalty='l2', max_iter=1, warm_start=True)
        model.classes_ = np.array([0, 1])
        model.coef_ = parameters[0]
        model.intercept_ = parameters[1]
        
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)
        loss = log_loss(y_test, y_prob, labels=[0, 1])
        acc = accuracy_score(y_test, y_pred)
        return float(loss), {"accuracy": float(acc)}
    return evaluate

class MockProxy(fl.server.client_proxy.ClientProxy):
    def __init__(self, cid, node_id=1):
        self.cid = cid
        self.node_id = node_id
    def get_properties(self, ins, timeout, group_id): pass
    def get_parameters(self, ins, timeout, group_id): pass
    def fit(self, ins, timeout, group_id): pass
    def evaluate(self, ins, timeout, group_id): pass
    def reconnect(self, ins, timeout): pass

def run_simulation(num_clients=3, num_rounds=5, privacy_multiplier=0.01, log_path='metrics.json'):
    # Generate data
    client_data, (X_test, y_test) = generate_synthetic_healthcare_data(n_samples=5000, n_clients=num_clients)
    
    # Check if we should clear metrics
    if os.path.exists(log_path):
        os.remove(log_path)
    with open(log_path, 'w') as f:
        json.dump([], f)

    eval_fn = get_evaluate_fn(X_test, y_test)

    # Strategy setup
    strategy = DPFedAvg(
        privacy_multiplier=privacy_multiplier,
        log_path=log_path,
        eval_fn=eval_fn,
    )

    # Initialize clients
    clients = []
    for cid in range(num_clients):
        X_train, y_train = client_data[str(cid)]
        clients.append(HospitalClient(str(cid), X_train, y_train))
        
    # Init initial parameters (NDArrays)
    dummy_model = SGDClassifier(loss='log_loss', max_iter=1, warm_start=True)
    dummy_model.classes_ = np.array([0, 1])
    dummy_model.coef_ = np.zeros((1, X_test.shape[1]))
    dummy_model.intercept_ = np.zeros((1,))
    
    current_ndarrays = [dummy_model.coef_, dummy_model.intercept_]
    
    for round_num in range(1, num_rounds + 1):
        # 1. FIT ROUND
        fit_results = []
        for c in clients:
            # Local training
            res_ndarrays, num_examples, metrics = c.fit(current_ndarrays, config={"local_epochs": 1})
            
            # Wrap to match Flower's aggregate_fit signature
            fit_res = fl.common.FitRes(
                status=fl.common.Status(code=fl.common.Code.OK, message=""),
                parameters=fl.common.ndarrays_to_parameters(res_ndarrays),
                num_examples=num_examples,
                metrics=metrics
            )
            fit_results.append((MockProxy(cid=c.cid, node_id=1), fit_res))
            
        # 2. AGGREGATE
        agg_params, _ = strategy.aggregate_fit(round_num, fit_results, [])
        if agg_params is not None:
            # We got updated parameters, including DP noise applied inside DPFedAvg
            current_ndarrays = fl.common.parameters_to_ndarrays(agg_params)
            
        # 3. EVALUATE (Centralized)
        loss, metrics = eval_fn(round_num, current_ndarrays, {})
        
        # Log to file, merging with what DPFedAvg wrote
        if os.path.exists(log_path):
            with open(log_path, 'r') as f:
                history = json.load(f)
                
            for h in history:
                if h["round"] == round_num:
                    h["global_accuracy"] = float(metrics["accuracy"])
            
            with open(log_path, 'w') as f:
                json.dump(history, f)

if __name__ == "__main__":
    num_clients = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    num_rounds = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    privacy_multiplier = float(sys.argv[3]) if len(sys.argv) > 3 else 0.01
    
    run_simulation(num_clients, num_rounds, privacy_multiplier)
