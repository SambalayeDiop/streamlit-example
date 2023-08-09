from collections import namedtuple
import altair as alt
import math
import streamlit as st
import pandas as pd
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve
from sklearn.metrics import precision_score, recall_score
st.set_option("deprecation.showPyplotGlobalUse", False)

def main():
    st.title("Application de Machine Learning pour la détection de fraude")
    st.subheader("Auteur: M DIOP")

    # Définition de la fonction pour télécharger les données
    @st.cache(persist=True)  # Pour éviter un changement d'état de calcul
    def load_data():
        data = pd.read_csv("creditcard.csv")
        return data

    # Affichage du dataset
    df = load_data()
    df_simple = df.sample(100)
    # On permet à l'utilisateur de l'afficher s'il désire
    if st.sidebar.checkbox('Afficher la base de données', False):
        st.subheader("Quelques données du dataset : Échantillon de 100 lignes")
        st.write(df_simple)

    seed = 123
    # Création de notre jeu d'entraînement et de test
    @st.cache(persist=True)
    def split_data(df):
        y = df['Class']
        x = df.drop('Class', axis=1)

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=seed)
        return x_train, x_test, y_train, y_test

    x_train, x_test, y_train, y_test = split_data(df)

    classifier = st.sidebar.selectbox(
        "Classificateur", ("Random Forest", "SVM", "Logistic Regression")
    )

    # Analyse de performance
    def plot_perf(graphes):
        if 'Confusion matrix' in graphes:
            st.subheader("Matrice de confusion")
            plot_confusion_matrix(model, x_test, y_test)
            st.pyplot()

        if 'ROC Curve' in graphes:
            st.subheader("Courbe ROC")
            plot_roc_curve(model, x_test, y_test)
            st.pyplot()

        if 'Precision Recall' in graphes:
            st.subheader("Courbe de précision Recall")
            plot_precision_recall_curve(model, x_test, y_test)
            st.pyplot()

    # Random Forest
    if classifier == "Random Forest":
        st.sidebar.subheader("Les hyperparamètres du modèle")
        n_arbres = st.sidebar.number_input("Nombre d'arbres pour le modèle de forêt", 100, 1000, step=10)
        profondeur_arbre = st.sidebar.number_input("La profondeur max du modèle de forêt", 1, 20, step=1)
        bootstrap = st.sidebar.radio("Échantillons bootstrap lors de la création d'arbres", (True, False))

        graphes_perf = st.sidebar.multiselect("Choisir un graphe de performance du modèle ML",
                                             ("Confusion matrix", "ROC Curve", "Precision Recall"))

    if st.sidebar.button("Exécuter", key="classify"):
        st.subheader("Random Forest Résultat")
        # Instanciation des classes du modèle
        model = RandomForestClassifier(n_estimators=n_arbres, max_depth=profondeur_arbre, bootstrap=bootstrap)
        model.fit(x_train, y_train)  # Entraînement du modèle

        # Prédiction du modèle
        y_pred = model.predict(x_test)

        # Métriques du modèle
        accuracy = model.score(x_test, y_test).round(3)
        precision = precision_score(y_test, y_pred).round(3)
        recall = recall_score(y_test, y_pred).round(3)

        # Afficher les métriques
        st.write("Accuracy :", accuracy)
        st.write("Precision :", precision)
        st.write("Recall :", recall)

        # Afficher les graphiques de performances
        plot_perf(graphes_perf)


if __name__ == '__main__':
    main()
