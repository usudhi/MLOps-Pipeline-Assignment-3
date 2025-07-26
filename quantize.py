import joblib
import numpy as np
import os
import torch
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import train

# 1. Ensure the scikit-learn model is trained and available
print("Running train.py to ensure model exists...")
train.train_and_save_model()

# 2. Load the scikit-learn model
sklearn_model = joblib.load('linear_regression_model.joblib')

# 3. Extract its learned parameters
coef = sklearn_model.coef_
intercept = sklearn_model.intercept_

# 4. Store unquantized parameters and save
unquantized_params = {'coef': coef, 'intercept': intercept}
joblib.dump(unquantized_params, 'unquant_params.joblib')

# 5. Perform manual quantization using a robust, PyTorch-like formula
def quantize_value(value):
    """Manually quantizes a numpy array to uint8 using a robust formula."""
    # Define quantization range for unsigned 8-bit integers
    qmin, qmax = 0, 255
    
    max_val, min_val = np.max(value), np.min(value)
    
    # Handle the case where all values are the same (e.g., the intercept)
    if max_val == min_val:
        scale = 0.0  # Use scale=0 as a flag for a constant value
        zero_point = qmin # Can be any value in range, doesn't matter
        quantized_value = np.full(value.shape, qmin, dtype=np.uint8)
        # Return the constant value itself for reconstruction
        return quantized_value, scale, min_val

    # Calculate scale
    scale = (max_val - min_val) / (qmax - qmin)
    
    # Calculate zero_point based on the formula: zero_point = qmin - round(min_val / scale)
    zero_point = qmin - np.round(min_val / scale)
    # Clamp the zero_point to be within the valid quantization range
    zero_point = np.clip(zero_point, qmin, qmax)
    
    # Quantize the value: q = round(r / scale) + z
    quantized_value = np.round(value / scale) + zero_point
    
    # Clip the final quantized values to the quantization range
    quantized_value = np.clip(quantized_value, qmin, qmax).astype(np.uint8)
    
    return quantized_value, scale, zero_point

quantized_coef, scale_coef, zp_coef = quantize_value(coef)
# For the intercept, the constant value case will be triggered
quantized_intercept, scale_intercept, zp_intercept_or_min_val = quantize_value(np.array([intercept]))

# 6. Store the simplified quantized parameters
quantized_params_dict = {
    'coef': quantized_coef, 'scale_coef': scale_coef, 'zp_coef': zp_coef,
    'intercept': quantized_intercept, 'scale_intercept': scale_intercept, 'zp_intercept_or_min_val': zp_intercept_or_min_val
}
joblib.dump(quantized_params_dict, 'quant_params.joblib')

# 7. Create a single-layer PyTorch model
class PyTorchLR(torch.nn.Module):
    def __init__(self, n_features):
        super(PyTorchLR, self).__init__()
        self.linear = torch.nn.Linear(n_features, 1)

    def forward(self, x):
        return self.linear(x)

# 8. De-quantize weights using the corresponding formula
def dequantize_value(quant_val, scale, zero_point_or_min_val):
    """De-quantizes a value from uint8."""
    # Handle the constant value case where scale is 0
    if scale == 0.0:
        return np.full(quant_val.shape, zero_point_or_min_val, dtype=np.float32)
        
    # Standard dequantization formula: r = (q - z) * s
    zero_point = zero_point_or_min_val
    return (quant_val.astype(np.float32) - zero_point) * scale

dequantized_coef = dequantize_value(quantized_coef, scale_coef, zp_coef)
dequantized_intercept = dequantize_value(quantized_intercept, scale_intercept, zp_intercept_or_min_val)[0]

# Initialize PyTorch model and set its weights manually
n_features = sklearn_model.n_features_in_
pytorch_model = PyTorchLR(n_features)
pytorch_model.linear.weight.data = torch.from_numpy(dequantized_coef).float().unsqueeze(0)
pytorch_model.linear.bias.data = torch.from_numpy(np.array([dequantized_intercept])).float()

# 9. Analyze and Report
housing = fetch_california_housing()
_, X_test, _, y_test = train_test_split(housing.data, housing.target, test_size=0.2, random_state=42)

# R² for original scikit-learn model
original_r2 = sklearn_model.score(X_test, y_test)

# R² for PyTorch model with de-quantized weights
X_test_tensor = torch.from_numpy(X_test).float()
with torch.no_grad():
    quantized_model_pred = pytorch_model(X_test_tensor).numpy().flatten()
quantized_r2 = r2_score(y_test, quantized_model_pred)

unquant_size_kb = os.path.getsize('unquant_params.joblib') / 1024
quant_size_kb = os.path.getsize('quant_params.joblib') / 1024

# Print the required comparison table
print("\n--- Model Comparison Analysis ---")
print(f"| {'Metric':<20} | {'Original Sklearn Model':<28} | {'Quantized Model':<28} |")
print(f"|----------------------|------------------------------|------------------------------|")
print(f"| R² Score             | {original_r2:<28.4f} | {quantized_r2:<28.4f} |")
print(f"| Model Size (KB)      | {unquant_size_kb:<28.2f} | {quant_size_kb:<28.2f} |")
