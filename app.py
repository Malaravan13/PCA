import gradio as gr
import pandas as pd
import pickle

# 🔹 Load your saved Random Forest model + preprocessing
with open("rf_model.sav", "rb") as f:
    rf_model, scaler, imputer, pca = pickle.load(f)

# 🔹 Function to predict from uploaded CSV
def predict_from_csv(file):
    # Read uploaded CSV (make sure it has same columns as training data)
    new_data = pd.read_csv(file.name)

    # Apply preprocessing → scaling → PCA
    new_imputed = imputer.transform(new_data)
    new_scaled = scaler.transform(new_imputed)
    new_pca = pca.transform(new_scaled)

    # Predict
    predictions = rf_model.predict(new_pca)

    # Attach predictions to dataframe
    new_data["Predicted_ViolentCrimesPerPop"] = predictions
    return new_data

# 🔹 Gradio interface
iface = gr.Interface(
    fn=predict_from_csv,
    inputs=gr.File(file_types=[".csv"], label="Upload CSV row(s)"),
    outputs="dataframe",
    title="Crime Rate Prediction with Random Forest + PCA",
    description="Upload a CSV with feature columns. The app preprocesses, applies PCA, and predicts violent crime rate."
)

if __name__ == "__main__":
    iface.launch()
