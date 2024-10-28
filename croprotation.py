import streamlit as st
import numpy as np
import pandas as pd
import joblib
import random
import streamlit.components.v1 as components
from deap import base, creator, tools, algorithms

# page config
st.set_page_config(page_title="Crop Rotation Planner", page_icon="🌿", layout='wide', initial_sidebar_state="expanded")

# pre-trained model and LabelEncoder
rf_classifier = joblib.load('random_forest.pkl')
label_encoder = joblib.load('label_encoder.pkl')

# dataset
data = pd.read_csv('soil.impact.csv')

# imágenes
crop_images = {
    'Arugula': 'images/Arugula.jpg',
    'Asparagus': 'images/Asparagus.jpg',
    'Beet': 'images/Beet.jpg',
    'Broccoli': 'images/Broccoli.jpg',
    'Cabbage': 'images/Cabbage.jpg',
    'Cauliflowers': 'images/Cauliflowers.jpg',
    'Chard': 'images/Chard.jpg',
    'Chilli Peppers': 'images/Chilli Peppers.jpg',
    'Cress': 'images/Cress.jpg',
    'Cucumbers': 'images/Cucumbers.jpg',
    'Eggplants': 'images/Eggplants.jpg',
    'Endive': 'images/Endive.jpg',
    'Grapes': 'images/Grapes.jpg',
    'Green Peas': 'images/Green Peas.jpg',
    'Kale': 'images/Kale.jpg',
    'Lettuce': 'images/Lettuce.jpg',
    'Potatoes': 'images/Potatoes.jpg',
    'Radicchio': 'images/Radicchio.jpg',
    'Spinach': 'images/Spinach.jpg',
    'Strawberry': 'images/Strawberry.jpg',
    'Tomatoes': 'images/Tomatoes.jpg',
    'Watermelon': 'images/Watermelon.jpg',
}

# function to predict top crops
def predict_top_crops(input_features, model, label_encoder, top_n=3):
    probabilities = model.predict_proba([input_features])[0]
    top_indices = np.argsort(probabilities)[-top_n:][::-1]
    top_crops = label_encoder.inverse_transform(top_indices)
    return top_crops

# GENETIC ALGORITHM
impact_scores = {'restorative': 1, 'neutral': 0, 'depleting': -1}
data['Impact_Score'] = data['Impact'].map(impact_scores)
impact_data = data[['Name', 'Impact', 'Soil_Type']].drop_duplicates()

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()

def filter_crops_by_soil(soil_type):
    return impact_data[impact_data['Soil_Type'] == soil_type]

def create_individual(soil_type, num_periods, periods_per_year):
    filtered_crops = filter_crops_by_soil(soil_type).values.tolist()
    return creator.Individual([random.choice(filtered_crops) for _ in range(num_periods)])

def evaluate(individual, periods_per_year):
    periods_per_crop = len(individual) // periods_per_year
    diversity_penalty = 0
    for i in range(0, len(individual), periods_per_crop):
        year_crops = [crop[0] for crop in individual[i:i+periods_per_crop]]
        unique_crops = len(set(year_crops))
        if unique_crops < periods_per_crop:
            diversity_penalty += (periods_per_crop - unique_crops) * 5
    unique_crops = len(set(crop[0] for crop in individual))
    impact_score = sum(impact_scores[crop[1]] for crop in individual)
    random_yield = sum(random.uniform(10, 30) for _ in individual)
    return unique_crops + impact_score + random_yield - diversity_penalty,

# all Streamlit interface 

st.markdown(f"""
<style>
    button{{
        color: white !important;
        border-color: #2E8B57 !important; 
    }}

    button:hover {{
        border: 7px ridge;
        background-color: #0e1117 !important;
    }}
</style>
""", unsafe_allow_html=True)


header_col1, header_col2 = st.columns([3, 6])
with header_col1:
    st.image('https://www.omdena.com/images/omdena.png', width=250)
with header_col2:
    st.title('Optimizing Crop Rotation for Small-scale Farmers in Rwanda')

# Layout for inputs and controls


col1, col2, col3 = st.columns([2.8, 0.5, 6])
with col1:
    st.header("Crop Recommendation System")

    # two nested columns within the existing column for input controls
    col1a, col1b = st.columns(2)

     # input to the first nested column
    with col1a:
        temperature = st.slider("Temperature (°C)", 10, 40, 24, 1)
        rainfall = st.slider("Rainfall (mm)", 400, 2500, 1000, 50)
        light_intensity = st.slider("Light_Intensity", 70, 1000, 500, 50)

      # input controls to the second nested column
    with col1b:
        nitrogen = st.number_input("Nitrogen Content (mg/ha)", 0, 400, 120, 10)
        phosphorus = st.number_input("Phosphorus Content (mg/ha)", 0, 360, 100, 10)
        potassium = st.number_input("Potassium Content (mg/ha)", 0, 570, 150, 10)
        season = st.selectbox("Season", ("Spring", "Summer", "Autumn", "Winter"))

    # season into one-hot encoding
    season_encoded = [0, 0, 0, 0]  # [Spring, Summer, Autumn, Winter]
    if season == "Spring":
        season_encoded = [1, 0, 0, 0]
    elif season == "Summer":
        season_encoded = [0, 1, 0, 0]
    elif season == "Autumn":
        season_encoded = [0, 0, 1, 0]
    elif season == "Winter":
        season_encoded = [0, 0, 0, 1]

    input_features = [temperature, rainfall, light_intensity, nitrogen, phosphorus, potassium] + season_encoded

    if st.button("Predict Top Crops", key='predict_crops'):
        top_crops = predict_top_crops(input_features, rf_classifier, label_encoder)
        st.session_state['top_crops'] = top_crops

    # GA interface in col1 continued
    st.header("Crop Rotation Plan")
    years = st.slider('Select the number of years for the crop rotation plan:', 2, 5, key='years_slider')
    crops_per_year = st.slider('Select the number of crops per year:', 1, 4, key='crops_per_year_slider')
    num_periods = years * crops_per_year
    periods_per_year = crops_per_year

    soil_types = data['Soil_Type'].unique()
    selected_soil_type = st.selectbox('Select soil type:', soil_types, key='soil_type_selectbox')

    toolbox.register("individual", create_individual, soil_type=selected_soil_type, num_periods=num_periods, periods_per_year=periods_per_year)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.4)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("evaluate", evaluate, periods_per_year=periods_per_year)

    if st.button('Generate Crop Rotation Plan'):
        population = toolbox.population(n=8000)
        result_population, log = algorithms.eaSimple(population, toolbox, cxpb=0.5, mutpb=0.1, ngen=30, verbose=True)
        best_individual = tools.selBest(result_population, 1)[0]
        result = pd.DataFrame(best_individual, columns=['Name', 'Impact', 'Soil_Type'])
        st.session_state['crop_rotation_result'] = result

with col3:
    if 'top_crops' in st.session_state:
        st.markdown("""
        <h1 style='
        text-align: center;
        padding: 40px;
        color: #2E8B57;
        font-family: Arial, sans-serif;
        '> 
          Top 3 Crops for actual conditions:
        </h1>
    """, unsafe_allow_html=True)
        cols = st.columns(len(st.session_state['top_crops']))
        for idx, crop in enumerate(st.session_state['top_crops']):
            with cols[idx]:
                st.image(crop_images.get(crop, 'images/default.jpg'), width=300, caption=crop)

    if 'crop_rotation_result' in st.session_state:
        st.markdown("<h1 style='text-align: center;  padding: 80px; color: #2E8B57; font-family: Arial, sans-serif; '>Best Crop Rotation Plan:</h1>", unsafe_allow_html=True)
        st.dataframe(st.session_state['crop_rotation_result'],  width=600)


# color: #2E8B57; green    #2c39b1-Omdena col
#blue hover button #3C39A2