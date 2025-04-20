import sys
import numpy as np
import joblib


model = joblib.load('model_for_pred.pkl')
scaler = joblib.load('scaled_for_pred.pkl')
input_features = np.array(sys.argv[1:], dtype=float)

if input_features.shape[0] == 7:
    input_features = np.insert(input_features, 0, 0)


scaled_features = scaler.transform(input_features.reshape(1, -1))

predict_val = model.predict(scaled_features)


print(f"{predict_val}")