import numpy as np
from model import ThreeLayerNet


def test(model_file, X_test, y_test):
    data = np.load(model_file)
    params = {key: data[key] for key in data.files if key not in ['hidden_size', 'activation']}
    hidden_size = data['hidden_size'].item()
    if 'activation' in data:
        activation = data['activation'].item()
    
    input_dim = X_test.shape[1]
    model = ThreeLayerNet(input_dim, hidden_size, 10)
    model.params = params
    model.eval()
    probs = model.forward(X_test)
    predictions = np.argmax(probs, axis=1)
    accuracy = np.mean(predictions == y_test)
    print(f"Test Accuracy: {accuracy:.4f}")