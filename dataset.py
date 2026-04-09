import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def generate_synthetic_healthcare_data(n_samples=5000, n_clients=3):
    """
    Generates tabular synthetic patient data and partitions it among clients (hospitals).
    """
    np.random.seed(42)
    # Features: Age, Blood Pressure, Heart Rate, BMI, Symptom_1, Symptom_2, Symptom_3
    ages = np.random.normal(55, 15, n_samples)
    bps = np.random.normal(120, 20, n_samples)
    hrs = np.random.normal(70, 10, n_samples)
    bmis = np.random.normal(25, 5, n_samples)
    sym1 = np.random.binomial(1, 0.4, n_samples)
    sym2 = np.random.binomial(1, 0.3, n_samples)
    sym3 = np.random.binomial(1, 0.1, n_samples)
    
    # Synthesize disease risk (target) based on feature combinations + noise
    logit = 0.05*(ages-55) + 0.02*(bps-120) + 0.1*(bmis-25) + 1.2*sym1 + 0.8*sym2 + 2.0*sym3 - 1.0
    probs = 1 / (1 + np.exp(-logit))
    y = np.random.binomial(1, probs)
    
    X = np.column_stack((ages, bps, hrs, bmis, sym1, sym2, sym3))
    
    # Scale continuous features
    scaler = StandardScaler()
    X[:, :4] = scaler.fit_transform(X[:, :4])
    
    # Train / global test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Partition across hospital clients
    client_data = {}
    chunk_size = len(X_train) // n_clients
    for i in range(n_clients):
        start = i * chunk_size
        end = start + chunk_size if i < n_clients - 1 else len(X_train)
        client_data[str(i)] = (X_train[start:end], y_train[start:end])
        
    return client_data, (X_test, y_test)
