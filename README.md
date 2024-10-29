# Optimizing Crop Rotation for Small-scale Farmers in Rwanda
![image](https://github.com/user-attachments/assets/873cd3e5-af9a-4b9b-a288-3b3a19cb2d20)

[Check the STREAMLIT APP](https://crop-recommendation-and-rotation-plan.streamlit.app/)

## Overview
This project focuses on developing an optimized crop rotation system for small-scale farmers in Rwanda, using a combination of data-driven analysis and machine learning models to make informed crop recommendations an finaly to create a crop rotation plan. This approach is intended to improve soil health, increase crop yield, and support sustainable agricultural practices.

### Dataset Overview
The dataset `soil.impact.csv` provides information on various crops along with their environmental and soil requirements. The dataset includes columns such as:
- **Name**: Crop name (e.g., Strawberry, Potato)
- **Fertility**: Fertility level required for crop growth
- **Temperature**: Temperature (°C) associated with optimal crop growth
- **Rainfall**: Rainfall levels (mm) suitable for crop growth
- **pH**: Soil pH requirements
- **Light Hours/Intensity**: Required daily sunlight hours and light intensity
- **Rh**: Relative Humidity (%)
- **Nitrogen, Phosphorus, Potassium**: Key soil nutrients
- **Yield**: Expected crop yield
- **Soil Type**: Type of soil best suited for each crop (e.g., Loam)
- **Season**: Recommended growing season (e.g., Summer, Spring)
- **Impact**: Soil impact status (e.g., "depleting" indicating nutrient depletion)

### Notebook Structure
1. **Data Loading and Initial Exploration**
    - Import libraries and load the soil impact dataset for initial data exploration.
    - Use `skimpy` to provide a summary overview of the data distribution and feature importance.

2. **Data Preprocessing**
    - Preprocessing steps include label encoding, scaling, and train-test splitting to prepare for model training.
    - Key libraries used: `pandas`, `numpy`, and `scikit-learn`.

3. **Modeling for Crop Recommendation**
    - Implement multiple machine learning models to predict crop suitability based on soil characteristics and environmental factors:
        - `GradientBoostingClassifier`
        - `RandomForestClassifier`
        - `LogisticRegression`
        - `XGBoost (XGBClassifier)`
    - Evaluation includes confusion matrix and LIME explanations to interpret model decisions.

4. **Exploratory Data Analysis (EDA)**
    - Visualizations and statistical summaries to understand correlations and distributions.

### Objectives
- Recommend suitable crops for rotation based on soil health and environmental conditions.
- Interpret model outputs to ensure explainability for farmers and agricultural stakeholders.
- Provide insights for a sustainable crop rotation plan that improves long-term soil quality.
