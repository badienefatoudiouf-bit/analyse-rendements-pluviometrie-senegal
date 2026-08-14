import gradio as gr
import joblib
import pandas as pd

model = joblib.load("modele_rendement.pkl")

colonnes = ['Rainfall_mm', 'Temperature_Celsius', 'Fertilizer_Used', 'Irrigation_Used',
            'Days_to_Harvest', 'Region_East', 'Region_North', 'Region_South', 'Region_West',
            'Soil_Type_Chalky', 'Soil_Type_Clay', 'Soil_Type_Loam', 'Soil_Type_Peaty',
            'Soil_Type_Sandy', 'Soil_Type_Silt', 'Crop_Barley', 'Crop_Cotton', 'Crop_Maize',
            'Crop_Rice', 'Crop_Soybean', 'Crop_Wheat', 'Weather_Condition_Cloudy',
            'Weather_Condition_Rainy', 'Weather_Condition_Sunny']

def predire_rendement(rainfall, temperature, fertilizer, irrigation, days_to_harvest,
                       region, soil_type, crop, weather):
    ligne = {col: 0 for col in colonnes}
    ligne['Rainfall_mm'] = rainfall
    ligne['Temperature_Celsius'] = temperature
    ligne['Fertilizer_Used'] = int(fertilizer)
    ligne['Irrigation_Used'] = int(irrigation)
    ligne['Days_to_Harvest'] = days_to_harvest
    ligne[f'Region_{region}'] = 1
    ligne[f'Soil_Type_{soil_type}'] = 1
    ligne[f'Crop_{crop}'] = 1
    ligne[f'Weather_Condition_{weather}'] = 1

    X_input = pd.DataFrame([ligne])[colonnes]
    prediction = model.predict(X_input)[0]
    return f"{prediction:.2f} tonnes/hectare"

interface = gr.Interface(
    fn=predire_rendement,
    inputs=[
        gr.Number(label="Précipitations (mm)", value=500),
        gr.Number(label="Température (°C)", value=25),
        gr.Checkbox(label="Engrais utilisé"),
        gr.Checkbox(label="Irrigation utilisée"),
        gr.Number(label="Jours jusqu'à récolte", value=120),
        gr.Dropdown(["East", "North", "South", "West"], label="Région"),
        gr.Dropdown(["Chalky", "Clay", "Loam", "Peaty", "Sandy", "Silt"], label="Type de sol"),
        gr.Dropdown(["Barley", "Cotton", "Maize", "Rice", "Soybean", "Wheat"], label="Culture"),
        gr.Dropdown(["Cloudy", "Rainy", "Sunny"], label="Météo"),
    ],
    outputs=gr.Textbox(label="Rendement prédit"),
    title="Prédicteur de rendement agricole",
    description="Modèle de régression linéaire entraîné sur 1M d'observations (dataset Kaggle)"
)

interface.launch(server_name="0.0.0.0", server_port=7860)