import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

# Title
st.title("Web App Using Streamlit")

# Image
st.image("Heart-Attack-blog.jpg", width=500)

# Dataset
st.title("Heart Attack Dataset")
data = pd.read_csv("heart_attack_dataset.csv")
st.write("Shape of the dataset:", data.shape)

# Sidebar menu
menu = st.sidebar.radio("Menu", ["Home", "Graphs"])

if menu == "Home":
    st.image("heart-attack-the-golden-hour-1.jpg", width=550)
    
    st.header("Tabular Data of the Dataset")
    if st.checkbox("Show Tabular Data"):
        st.table(data.head(150))
    
    st.header("Statistical Summary of DataFrame")
    if st.checkbox("Show Statistics"):
        st.table(data.describe())
    
    st.header("Correlation Graph")
    if st.checkbox("Show Correlation Graph"):
        # Filter numeric columns only
        numeric_data = data.select_dtypes(include=[np.number])
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(numeric_data.corr(), annot=True, cmap="coolwarm")
        st.pyplot(fig)

elif menu == "Graphs":
    st.title("Graphs")
    graph_type = st.selectbox("Select Type of Graph", ["Scatter Plot", "Bar Graph", "Histogram"])

    if graph_type == "Scatter Plot":
        if "Blood Pressure (mmHg)" in data.columns and "Treatment" in data.columns:
            value = st.slider("Filter Data by Age", 0, int(data["Age"].max()), 0)
            filtered_data = data[data["Age"] > value]
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.scatterplot(data=filtered_data, x="Age", y="Gender")
            st.pyplot(fig)
        else:
            st.write("The selected columns are not available in the dataset.")
    
    elif graph_type == "Bar Graph":
        if "Gender" in data.columns:
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.barplot(x="Gender", y=data.index, data=data)
            st.pyplot(fig)
        else:
            st.write("The selected column is not available in the dataset.")
    
    elif graph_type == "Histogram":
        if "Cholesterol (mg/dL)" in data.columns:
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.histplot(data["Cholesterol (mg/dL)"], kde=True)
            st.pyplot(fig)
        else:
            st.write("The selected column is not available in the dataset.")


