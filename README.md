# Optimizing Crop Rotation for Small-scale Farmers in Rwanda
![image](https://github.com/user-attachments/assets/873cd3e5-af9a-4b9b-a288-3b3a19cb2d20)

[Check the STREAMLIT APP](https://crop-recommendation-and-rotation-plan.streamlit.app/)

## Overview
This project focuses on developing an optimized crop rotation system for small-scale farmers in Rwanda, using a combination of data-driven analysis and machine learning models to make informed crop recommendations and finally to create a crop rotation plan. This approach is intended to improve soil health, increase crop yield, and support sustainable agricultural practices.

### Objectives
- Recommend crop rotations tailored to soil and climate conditions.  
- Enhance long-term soil quality while supporting higher yields and profitability.  
- Use genetic algorithms to design robust, sustainable crop rotation strategies.
- App development for practical use by stakeholders. 

  
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
- **Impact**: Soil impact status (e.g., "depleting" indicating nutrient depletion) - new column created in the Feature engineering part

### Notebook Structure
1. **Data Loading and Initial Exploration**
    - Libraries are imported, and the dataset is explored using tools like skimpy for summarization and distribution insights.

2. **Exploratory Data Analysis (EDA)**
    - Visualizations and statistical summaries to understand correlations and distributions.

3. **Data Preprocessing**
    - Preprocessing steps include label encoding, scaling, and train-test splitting to prepare for model training.
    - Key libraries used: `pandas`, `numpy`, and `scikit-learn`.

4. **Modeling for Crop Recommendation**
    - Implementing multiple machine learning models to predict crop suitability based on soil characteristics and environmental factors:
        - `GradientBoostingClassifier`
        - `RandomForestClassifier`
        - `LogisticRegression`
        - `XGBoost (XGBClassifier)`
    - Evaluation includes confusion matrix and feature importance to choose the final model.

5. **Genetic Algorithm for Crop Rotation Optimization**  
   - Implements a genetic algorithm to generate and evolve crop rotation plans.  
   - The algorithm optimizes rotations for 3-5 years, balancing crop yields, and soil health. With data reffering market it can jbe added profitability as a feature to (also) maximize.
   - Evaluates solutions based on constraints like nutrient cycling and environmental sustainability.


### Streamlit App 
 - Crop recommendation- user input: environmental and soil features.
 - Crop rotation with Genetic Algorithms for 2-5 year, 1-4 crops to grow and soil type.
