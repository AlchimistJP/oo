import streamlit as st
st.title('Ma premiere application')

# Sélecteur de texte
option = st.selectbox('Choisissez une option', ['Option 1', 'Option 2', 'Option 3'])
st.write('Vous avez sélectionné :', option)
# Curseur
valeur = st.slider('Sélectionnez une valeur', 0, 100, 50)
st.write('Valeur sélectionnée :', valeur)
# Affichage de données tabulaires
import pandas as pd
df = pd.DataFrame({
 'Nom': ['Alice', 'Bob', 'Charlie'],
 'Âge': [30, 35, 40]
 })
st.write(df)
# Affichage d'un graphique
import matplotlib.pyplot as plt
import numpy as np

# Changez le thème
st.markdown("""<style>body { background-color: #f0f0f0;} </style>""", unsafe_allow_html=True)
# Texte mis en forme
st.markdown('**Gras** :boom:')
st.write('Texte en _italique_.')

from sklearn.metrics import r2_score

# Générer des données aléatoires pour l'exemple
np.random.seed(0)
X_train = np.random.rand(100, 1) * 10
y_train = 3 * X_train ** 2 - 5 * X_train + 2 + np.random.randn(100, 1) * 5
# Entraîner le modèle
coefficients = np.polyfit(X_train.flatten(), y_train.flatten(), 2)
# Calculer le score du modèle
y_pred = np.polyval(coefficients, X_train.flatten())
r2=r2_score(y_train,y_pred)
#DéfinirlapageStreamlit
st.title('DéploiementdemodèleavecStreamlit')
#Widgetpoursaisirunevaleurd'entrée
input_value=st.slider('Valeurd\'entrée',min_value=0.0,max_value=10.0,step=0.1)
#Prédictionaveclemodèleentraîné
prediction=np.polyval(coefficients,input_value)
#Affichagedelaprédiction
st.write(f'Prédictionpourlavaleurd\'entrée{input_value}:{prediction}')
#Affichageduscoredumodèle
st.write(f'Scoredumodèle(R²):{r2:.4f}')
#Affichagedugraphiqueinteractif
st.subheader('Graphiqueinteractif')
fig,ax=plt.subplots()
ax.plot(X_train,y_train,'bo',label='Donnéesd\'entraînement')
x_values=np.linspace(0,10,100)
y_values=np.polyval(coefficients,x_values)
ax.plot(x_values,y_values,'r-',label='Modèleprédictif')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_title('Régressionquadratique')
ax.legend()
st.pyplot(fig)
