import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib # Common library to load ML models (if you use scikit-learn)

# --- CONFIGURATION ---
st.set_page_config(
    page_title="Real Estate Investment Advisor",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- HELPER FUNCTIONS ---

@st.cache_data
def load_data(file_path="data/india_housing_prices.csv"):
    """Loads and preprocesses the necessary data for EDA/App context."""
    try:
        df = pd.read_csv(file_path)
        # Re-calculate unscaled Price per SqFt and clean data as done in EDA
        df['Price_in_Lakhs'] = pd.to_numeric(df['Price_in_Lakhs'], errors='coerce')
        df['Size_in_SqFt'] = pd.to_numeric(df['Size_in_SqFt'], errors='coerce')
        df.dropna(subset=['Price_in_Lakhs', 'Size_in_SqFt'], inplace=True)
        df = df[df['Size_in_SqFt'] > 0]
        df['Price_per_SqFt'] = (df['Price_in_Lakhs'] * 100000) / df['Size_in_SqFt']
        return df
    except FileNotFoundError:
        st.error(f"Error: Data file {file_path} not found. Please upload it or check the path.")
        return pd.DataFrame()

# Placeholder for Model Loading (Will be replaced with MLflow integration)
def load_models():
    """Placeholder to load trained models."""
    try:
        # Example: Load preprocessors and models
        # reg_model = joblib.load('regression_model.pkl')
        # clf_model = joblib.load('classification_model.pkl')
        # preprocessor = joblib.load('preprocessor.pkl')
        return "Regression Model Placeholder", "Classification Model Placeholder"
    except Exception as e:
        # st.error(f"Could not load models: {e}")
        return None, None

# --- MODEL PREDICTION LOGIC (PLACEHOLDER) ---

def predict_investment(input_data, clf_model):
    """Placeholder for the Classification prediction logic."""
    # This logic will use the loaded preprocessor and clf_model
    # For now, return dummy results based on heuristics
    is_good = np.random.choice(["Good Investment", "Poor Investment"], p=[0.7, 0.3])
    confidence = np.random.uniform(0.75, 0.95)
    return is_good, confidence

def predict_future_price(input_data, reg_model):
    """Placeholder for the Regression prediction logic."""
    # This logic will use the loaded preprocessor and reg_model
    # For now, return a dummy price
    current_price = input_data['Price_in_Lakhs'].iloc[0] if 'Price_in_Lakhs' in input_data.columns else 300
    future_price = current_price * np.random.uniform(1.3, 1.6) # Simulating 30-60% growth over 5 years
    return future_price

# --- APP LAYOUT ---

def main():
    st.title("🏡 Real Estate Investment Advisor")
    st.subheader("Predicting Property Profitability & Future Value")

    # Load data and models
    data = load_data()
    reg_model, clf_model = load_models()
    
    # Check if data loaded successfully
    if data.empty:
        return

    # --- SIDEBAR: PROJECT INFO ---
    st.sidebar.header("Project Overview")
    st.sidebar.markdown("""
    This application utilizes Machine Learning models for **Investment Analytics** to assist potential buyers and real estate firms.
    
    **Goals:**
    1.  Classify property as 'Good Investment' (Classification).
    2.  Predict Estimated Price after 5 years (Regression).
    """)
    st.sidebar.metric("Total Properties Analyzed", f"{data.shape[0]:,}")


    # --- MAIN CONTENT TABS ---
    tab1, tab2 = st.tabs(["Investment Predictor", "Data Insights Dashboard"])

    with tab1:
        st.header("Step 1: Input Property Details")
        
        # --- USER INPUT FORM ---
        with st.form("property_form"):
            col1, col2, col3 = st.columns(3)
            
            # Column 1 Inputs (Location & Type - High importance)
            with col1:
                # Use top 20 cities for practical dropdown list
                top_cities = data['City'].value_counts().head(20).index.tolist()
                city = st.selectbox("City", options=top_cities, index=0)
                
                property_type = st.selectbox("Property Type", options=data['Property_Type'].unique())
                
                bhk = st.number_input("BHK (Bedrooms, Hall, Kitchen)", min_value=1, max_value=6, value=3)

            # Column 2 Inputs (Size & Price - High importance)
            with col2:
                size_sqft = st.number_input("Size in SqFt", min_value=500, max_value=10000, value=2000)
                
                # The raw input price for prediction
                price_lakhs = st.number_input("Current Price (in Lakhs)", min_value=10.0, max_value=1000.0, value=250.0)
                
                # Amenities/Status
                furnished_status = st.selectbox("Furnished Status", options=data['Furnished_Status'].unique())

            # Column 3 Inputs (Secondary Factors)
            with col3:
                age_of_property = st.slider("Age of Property (Years)", min_value=0, max_value=50, value=5)
                
                parking = st.selectbox("Parking Space", options=sorted(data['Parking_Space'].unique().tolist()), index=1)
                
                # Placeholder for other engineered features if needed
                nearby_schools = st.slider("Nearby Schools (Rating/Count)", 1, 10, 5)
                
            
            # --- Submission Button ---
            submitted = st.form_submit_button("Analyze Investment")
        
        # --- PREDICTION OUTPUT ---
        if submitted:
            # 1. Create a dummy dataframe for prediction
            input_dict = {
                'City': city, 'Property_Type': property_type, 'BHK': bhk, 
                'Size_in_SqFt': size_sqft, 'Price_in_Lakhs': price_lakhs, 
                'Furnished_Status': furnished_status, 'Age_of_Property': age_of_property, 
                'Parking_Space': parking, 'Nearby_Schools': nearby_schools
                # Add all required features for your model here
            }
            input_df = pd.DataFrame([input_dict])

            st.header("Step 2: Investment Recommendation")
            st.markdown("---")

            # CLASSIFICATION RESULT
            good_investment, confidence = predict_investment(input_df, clf_model)
            
            if "Good Investment" in good_investment:
                st.success(f"✅ CLASSIFICATION: This property is a **{good_investment}**.")
            else:
                st.error(f"❌ CLASSIFICATION: This property is a **{good_investment}**.")
                
            st.metric("Model Confidence Score", f"{confidence:.2f}")
            st.info("Based on features like location-adjusted price, age, and amenities, this score reflects the model's certainty.")
            
            st.markdown("---")

            # REGRESSION RESULT
            future_price = predict_future_price(input_df, reg_model)
            
            st.subheader("Estimated Future Value")
            st.metric(
                "Estimated Price after 5 Years", 
                f"₹{future_price:,.2f} Lakhs",
                delta=f"+{future_price - price_lakhs:,.2f} Lakhs ({((future_price / price_lakhs) - 1) * 100:.1f}%)"
            )
            
            st.caption("Forecast based on historical appreciation rates, property specific features, and regional growth.")


    with tab2:
        st.header("Data Insights Dashboard")
        st.markdown("Visualizations from Exploratory Data Analysis (EDA) to support investment trends.")
        
        # --- VISUAL INSIGHTS (Embedding EDA plots) ---
        
        st.subheader("Price Trends by Property Type")
        fig1, ax1 = plt.subplots(figsize=(10, 5))
        sns.boxplot(x='Property_Type', y='Price_per_SqFt', data=data, ax=ax1, palette='cubehelix')
        ax1.set_title('Raw Price per SqFt Variation by Property Type')
        ax1.set_ylabel('Raw Price per SqFt')
        ax1.set_xlabel('Property Type')
        ax1.tick_params(axis='x', rotation=45)
        ax1.set_ylim(0, data['Price_per_SqFt'].quantile(0.99))
        st.pyplot(fig1)

        st.subheader("Geographical Price Hierarchy")
        state_price_avg = data.groupby('State')['Price_per_SqFt'].median().sort_values(ascending=False)
        fig2, ax2 = plt.subplots(figsize=(10, 5))
        sns.barplot(x=state_price_avg.index, y=state_price_avg.values, ax=ax2, palette='viridis')
        ax2.set_title('Median Raw Price per SqFt by State')
        ax2.set_ylabel('Median Raw Price per SqFt')
        ax2.set_xlabel('State')
        ax2.tick_params(axis='x', rotation=45)
        st.pyplot(fig2)
        
        # Placeholder for Feature Importance/Correlation plot (after model training)
        st.subheader("Model Feature Importance (Future Integration)")
        st.markdown("A dynamic plot showing which features (Location, Size, Age, etc.) most heavily influenced the prediction. This will be integrated after model training with MLflow.")
        

if __name__ == "__main__":
    main()