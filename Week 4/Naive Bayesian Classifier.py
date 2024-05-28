# streamlit_app.py

import streamlit as st
import pandas as pd
from sklearn import tree
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Title for the Streamlit app
st.title("Tennis Match Prediction App")

# File uploader to load data
uploaded_file = st.file_uploader("Upload your tennis dataset (CSV file)", type="csv")

if uploaded_file is not None:
    # Read the data from CSV
    data = pd.read_csv(uploaded_file)
    st.write("The first 5 values of data are:")
    st.write(data.head())
    
    # Obtain Train data and Train output
    X = data.iloc[:,:-1]
    y = data.iloc[:,-1]

    # Convert categorical features to numerical values
    le_outlook = LabelEncoder()
    X.Outlook = le_outlook.fit_transform(X.Outlook)

    le_Temperature = LabelEncoder()
    X.Temperature = le_Temperature.fit_transform(X.Temperature)

    le_Humidity = LabelEncoder()
    X.Humidity = le_Humidity.fit_transform(X.Humidity)

    le_Windy = LabelEncoder()
    X.Windy = le_Windy.fit_transform(X.Windy)

    le_PlayTennis = LabelEncoder()
    y = le_PlayTennis.fit_transform(y)

    st.write("Now the Train data is:")
    st.write(X.head())

    st.write("Now the Train output is:")
    st.write(y)

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

    # Train the classifier
    classifier = GaussianNB()
    classifier.fit(X_train, y_train)

    # Display accuracy
    accuracy = accuracy_score(classifier.predict(X_test), y_test)
    st.write("Accuracy is:", accuracy)

    # Allow user to input new data for prediction
    st.write("Enter new data to make a prediction:")

    new_outlook = st.selectbox("Outlook", le_outlook.classes_)
    new_temperature = st.selectbox("Temperature", le_Temperature.classes_)
    new_humidity = st.selectbox("Humidity", le_Humidity.classes_)
    new_windy = st.selectbox("Windy", le_Windy.classes_)

    if st.button("Predict"):
        new_data = pd.DataFrame({
            'Outlook': [le_outlook.transform([new_outlook])[0]],
            'Temperature': [le_Temperature.transform([new_temperature])[0]],
            'Humidity': [le_Humidity.transform([new_humidity])[0]],
            'Windy': [le_Windy.transform([new_windy])[0]]
        })

        prediction = classifier.predict(new_data)
        prediction_label = le_PlayTennis.inverse_transform(prediction)
        st.write("Prediction:", prediction_label[0])

# To run the app, use the command `streamlit run streamlit_app.py`
streamlit run streamlit_app.py
