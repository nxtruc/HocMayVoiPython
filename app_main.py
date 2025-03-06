import streamlit as st
import Classification 
import Clustering_Algorithms 
import Linear_Regression 
import PCA_t_sne 

st.set_page_config(page_title="Machine Learning App", layout="wide")

menu = ["Classification", "Clustering", "Linear Regression", "PCA/t-SNE"]
choice = st.sidebar.selectbox("Chọn chức năng", menu)

if choice == "Classification":
    Classification.run()
elif choice == "Clustering":
    Clustering_Algorithms.run()
elif choice == "Linear Regression":
    Linear_Regression.run()
elif choice == "PCA/t-SNE":
    PCA_t_sne.run()
