#!/usr/bin/env python
# coding: utf-8

import streamlit as st
import joblib
import numpy as np
import pandas as pd
#fonction pour afficher la prediction
def main():
    def predict_species(user_data):
        prediction=model.predict(user_data.reshape(1,-1))[0]
       
        #Associer la valeur predite a la classe correspondante
        if prediction==0:
            return 'setosa'
        elif prediction==1:
            return 'versicolor'
        else:
            return 'virginica'
        
    # Afficher un titre
    st.title('Prédiction de l\'espèce d\'Iris')
       
    # Chargement du modèle
    model = joblib.load('iris_model.pkl')

    # Obtenir les données de l'Iris à prédire
    sepal_length = st.number_input('Longueur du sépale')
    sepal_width = st.number_input('Largeur du sépale')
    petal_length = st.number_input('Longueur du pétale')
    petal_width = st.number_input('Largeur du pétale')

    # Préparer les données
    user_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

    # Faire la prédiction
    prediction = predict_species(user_data)

    # Afficher la prédiction
    if st.button('Prédire l\'espèce'):
        st.write(f'L\'espèce prédite est : {prediction}')

if __name__ == '__main__':
    main()
