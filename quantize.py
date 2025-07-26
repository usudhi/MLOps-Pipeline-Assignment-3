import joblib
import numpy as np
import os
import torch
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import train

print("Running train.py to ensure model exists...")
train.train_and_save_model()

sklearn_model = joblib.load('linear_regression_model.joblib')
coef = sklearn_model.coef_
intercept = sklearn_model.intercept_

unquantized_params = {'coef': coef, 'intercept': intercept}
joblib.dump(unquantized_params, 'unquant_params.joblib')

def quantize_value(value):
    max_val, min_val = np.max(value), np.min(value)
    if max_val == min_val:
        scale = 1.0
        quantized_value = np.full(value.shape, 128, dtype=np.uint8)
        zero_point = 128 - max_val
        return quantized_value, scale, zero_point
    scale = (max_val - min_val) / 255.0
    zero_point = -min_val / scale
    quantized_value = np.round(value / scale + zero_point)
    quantized_value = np.clip(quantized_value, 0, 255).astype(np.uint8)
    return quantized_value, scale, zero_point

quantized_coef, scale_coef, zp_coef = quantize_value(coef)
quantized_intercept, scale_intercept, zp_intercept = quantize_value(np.array([intercept]))

quantized_params_dict = {
    'coef': quantized_coef, 'scale_coef': scale_coef, 'zp_coef': zp_coef,
    'intercept': quantized_intercept, 'scale_intercept': scale_intercept, 'zp_intercept': zp_intercept
}
joblib.dump(quantized_params_dict, 'quant_params.joblib')

class PyTorchLR(torch.nn.Module):
    def __init__(self, n_features):
        super(PyTorchLR, self).__init__()
        self.linear = torch.nn.Linear(n_features, 1)
    def forward(self, x):
        return self.linear(x)

def dequantize_value(quant_val, scale, zero_point):
    return (quant_val.astype(np.float32) - zero_point) * scale

dequantized_coef = dequantize_value(quantized_coef, scale_coef, zp_coef)
dequantized_intercept = dequantize_value(quantized_intercept, scale_intercept, zp_intercept)[0]

n_features = sklearn_model.n_features_in_
pytorch_model = PyTorchLR(n_features)
pytorch_model.linear.weight.data = torch.from_numpy(dequantized_coef).float().unsqueeze(0)
pytorch_model.linear.bias.data = torch.from_numpy(np.array([dequantized_intercept])).float()

housing = fetch_california_housing()
_, X_test, _, y_test = train_test_split(housing.data, housing.target, test_size=0.2, random_state=42)

original_r2 = sklearn_model.score(X_test, y_test)

X_test_tensor = torch.from_numpy(X_test).float()
with torch.no_grad():
    quantized_model_pred = pytorch_model(X_test_tensor).numpy().flatten()
quantized_r2 = r2_score(y_test, quantized_model_pred)

unquant_size_kb = os.path.getsize('unquant_params.joblib') / 1024
quant_size_kb = os.path.getsize('quant_params.joblib') / 1024

print("\n--- Model Comparison Analysis ---")
print(f"| {'Metric':<20} | {'Original Sklearn Model':<28} | {'Quantized Model':<28} |")
print(f"|----------------------|------------------------------|------------------------------|")
print(f"| R² Score             | {original_r2:<28.4f} | {quantized_r2:<28.4f} |")
print(f"| Model Size (KB)      | {unquant_size_kb:<28.2f} | {quant_size_kb:<28.2f} |")