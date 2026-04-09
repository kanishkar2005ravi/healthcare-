import flwr as fl
import numpy as np
import json
import os

class DPFedAvg(fl.server.strategy.FedAvg):
    def __init__(self, privacy_multiplier=0.01, log_path='metrics.json', eval_fn=None, *args, **kwargs):
        kwargs["evaluate_fn"] = eval_fn
        super().__init__(*args, **kwargs)
        self.privacy_multiplier = privacy_multiplier
        self.log_path = log_path
        self.history_log = []

    def aggregate_fit(self, server_round, results, failures):
        """Aggregate fit results and apply central differential privacy."""
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)
        
        if aggregated_parameters is not None:
            # Convert parameters back to numpy arrays
            params_list = fl.common.parameters_to_ndarrays(aggregated_parameters)
            
            # Apply Differential Privacy
            # In a real secure aggregation scenario, this happens under encryption.
            # Here we simulate the final sum having Gaussian noise calibrated to the privacy budget.
            noisy_params = []
            for param in params_list:
                noise = np.random.normal(loc=0.0, scale=self.privacy_multiplier, size=param.shape)
                noisy_params.append(param + noise)
            
            # Repackage the parameter object
            aggregated_parameters = fl.common.ndarrays_to_parameters(noisy_params)
            
            # Approximation of data leakage risk for dashboard
            data_leakage_risk = max(0.01, 1.0 - (self.privacy_multiplier * 50))
            if data_leakage_risk > 1.0:
                data_leakage_risk = 1.0
                
            # Clear log on first round
            if server_round == 1 and os.path.exists(self.log_path):
                os.remove(self.log_path)
                
            entry = {
                "round": server_round,
                "privacy_multiplier": self.privacy_multiplier,
                "leakage_risk": float(data_leakage_risk)
            }
            
            self.history_log.append(entry)
            with open(self.log_path, 'w') as f:
                json.dump(self.history_log, f)
                
        return aggregated_parameters, aggregated_metrics
    
    def aggregate_evaluate(self, server_round, results, failures):
        loss, metrics = super().aggregate_evaluate(server_round, results, failures)
        
        # Merge metrics into tracking file
        if metrics is not None and os.path.exists(self.log_path):
            with open(self.log_path, 'r') as f:
                history = json.load(f)
                
            for h in history:
                if h["round"] == server_round:
                    if "accuracy" in metrics:
                        h["global_accuracy"] = float(metrics["accuracy"])
            
            with open(self.log_path, 'w') as f:
                json.dump(history, f)
                
        return loss, metrics
