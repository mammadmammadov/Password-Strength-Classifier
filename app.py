import shap
import matplotlib
import matplotlib.pyplot as plt
from flask import Flask, render_template, request
import numpy as np
import pickle
import pandas as pd

matplotlib.use('Agg')

app = Flask(__name__)

with open('cat_boost_model.pkl', 'rb') as file:
    model = pickle.load(file)


def extract_features(password):
    length = len(password)
    lowercase_freq = len([char for char in password if char.islower()]) / length
    uppercase_freq = 1 if len([char for char in password if char.isupper()]) > 0 else 0
    digit_freq = len([char for char in password if char.isdigit()]) / length
    special_char_freq = len([char for char in password if not char.isalnum()]) / length
    return np.array([[length, lowercase_freq, uppercase_freq, digit_freq, special_char_freq]])


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        password = request.form['password']

        features = extract_features(password)

        result = model.predict(features)

        if result[0] == 0:
            prediction_text = "Weak"
            prediction_class = "weak"
        elif result[0] == 1:
            prediction_text = "Medium"
            prediction_class = "medium"
        else:
            prediction_text = "Strong"
            prediction_class = "strong"

        # Calculate SHAP values
        explainer = shap.Explainer(model)
        shap_values = explainer(pd.DataFrame(features,
                                             columns=["length", "lowercase_freq", "uppercase_freq", "digit_freq",
                                                      "special_char_freq"]))

        # Get SHAP values for the instance (for the first prediction)

        shap_values_for_instance = shap_values[0]
        print(f"Results: ${result}")
        print(shap_values_for_instance)

        # Get the predicted class index
        predicted_class_index = result[0]

        # Select the SHAP values for the predicted class (adjust to get a 1D array)
        shap_values_for_class = shap_values_for_instance.values[:,
                                predicted_class_index].flatten()  # Flatten to 1D array
        base_value_for_class = shap_values_for_instance.base_values[
            predicted_class_index]  # Base value for the predicted class
        feature_values = shap_values_for_instance.data  # Original feature values

        # Create the explanation for the predicted class
        explanation = shap.Explanation(
            values=shap_values_for_class,
            base_values=base_value_for_class,
            data=feature_values,
            feature_names=["length", "lowercase_freq", "uppercase_freq", "digit_freq", "special_char_freq"]
        )

        plt.figure(figsize=(12, 8))

        shap.plots.waterfall(explanation)  # Now passing the explanation for the predicted class
        shap_image_path = 'static/shap_plot.png'
        plt.savefig(shap_image_path, bbox_inches='tight')
        plt.close()

        print(f"SHAP Image Path: {shap_image_path}")

        return render_template('result.html', prediction=prediction_text, prediction_class=prediction_class,
                               shap_image_path=shap_image_path)


@app.route('/show_shap', methods=['POST'])
def show_shap():
    return render_template('shap_values.html')


if __name__ == '__main__':
    app.run(debug=True)
