# Importation des bibliothèques
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Chargement des données
data = pd.read_csv('Marine_Fish_Data.csv')

# Exploration initiale
print("Aperçu des données :")
print(data.head())

# Sélection des colonnes pertinentes
selected_features = ['Water_Temperature', 'Pollution_Level', 'Region']
data = data[selected_features]

# Vérification des valeurs manquantes
print("Valeurs manquantes par colonne :")
print(data.isnull().sum())

# Nettoyage des données (suppression des valeurs manquantes)
data = data.dropna()

# Conversion des données catégoriques (Region) en numériques
data['Region'] = data['Region'].astype('category').cat.codes

# Normalisation des données pour le clustering

scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)

# Clustering avec K-Means
kmeans = KMeans(n_clusters=3, random_state=42)
data['Cluster'] = kmeans.fit_predict(scaled_data)

# Visualisation des clusters
sns.scatterplot(x=data['Water_Temperature'], y=data['Pollution_Level'], hue=data['Cluster'])
plt.title('Clusters en fonction de la température et de la pollution')
plt.xlabel('Température de l\'eau')
plt.ylabel('Niveau de pollution')
plt.legend(title='Cluster')
plt.show()