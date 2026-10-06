import streamlit as st
import joblib
import pandas as pd

st.title("Used Car Price Prediction")

# Load trained model
model = joblib.load("used_car_price_xgb_model.pkl")

st.success("Model loaded successfully.")

st.header("Enter Car Details")

title = st.text_input("Car Title", "Audi A3")
registration_year = st.number_input(
    "Registration Year",
    min_value=1990,
    max_value=2023,
    value=2018
)

mileage = st.number_input(
    "Mileage (miles)",
    min_value=0,
    value=30000
)

previous_owners = st.number_input(
    "Previous Owners",
    min_value=0,
    value=1
)

fuel_type = st.selectbox(
    "Fuel Type",
    ["Petrol", "Diesel", "Hybrid", "Electric", "Other"]
)

body_type = st.selectbox(
    "Body Type",
    ["Hatchback", "Suv", "Saloon", "Estate", "Coupe", "Convertible", "Other"]
)

engine = st.text_input("Engine", "1.5L")

gearbox = st.selectbox(
    "Gearbox",
    ["Manual", "Automatic", "Semi-Automatic"]
)

emission_class = st.selectbox(
    "Emission Class",
    ["Euro 6", "Euro 5", "Euro 4", "Euro 3", "Other"]
)

service_history = st.selectbox(
    "Service History",
    ["Full", "Partial", "None"]
)

doors = st.number_input(
    "Doors",
    min_value=2,
    max_value=6,
    value=5
)

seats = st.number_input(
    "Seats",
    min_value=2,
    max_value=9,
    value=5
)

if st.button("Predict Car Price"):

    # Validate inputs
    if not title.strip():
        st.error("Please enter a car title.")
        st.stop()

    try:
        engine_litres = float(
            engine.upper().replace("L", "").strip()
        )

        if engine_litres <= 0:
            st.error("Engine size must be greater than 0.")
            st.stop()

    except ValueError:
        st.error("Please enter a valid engine size, for example: 1.5L")
        st.stop()

    # Feature engineering
    brand = title.strip().split()[0]
    car_age = 2023 - registration_year

    # Create input DataFrame
    input_data = pd.DataFrame({
    "Mileage(miles)": [mileage],
    "Registration_Year": [registration_year],
    "Previous Owners": [previous_owners],
    "Doors": [doors],
    "Seats": [seats],
    "Car_Age": [car_age],
    "Engine_Litres": [engine_litres],
    "Brand": [brand],
    "Fuel type": [fuel_type],
    "Body type": [body_type],
    "Gearbox": [gearbox],
    "Emission Class": [emission_class],
    "Service history": [service_history]
})

    prediction = model.predict(input_data)

    st.subheader("Prediction Result")

predicted_price = prediction[0]

st.metric(
    label="Estimated Car Price",
    value=f"£{predicted_price:,.2f}"
)
st.caption(
    "This price is an ML-based estimate and may differ from the actual market price."
)