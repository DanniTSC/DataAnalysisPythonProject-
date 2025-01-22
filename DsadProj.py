# Uniunea europeana media pe 2020	13.6	31.0	19.7
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
#setare nr cpu pentru multithread pentru joblib si scikit learn
import os
os.environ["LOKY_MAX_CPU_COUNT"] = "4"

#Obiectivul proiectului este să analizăm și să identificăm tipare în participarea țărilor europene
# la activitățile fizice, utilizând metode statistice și tehnici de învățare automată (clusterizare).

file_path = 'activitati_fizice_curat.csv'
data1 = pd.read_csv(file_path)

# 1. Curatare date
def clean_data(data):
    # Selectam doar coloanele numerice
    numeric_cols = data.select_dtypes(include=[np.number])
    # Inlocuim valorile lipsa cu media pe coloanele numerice
    data[numeric_cols.columns] = numeric_cols.fillna(numeric_cols.mean())
    print("Valorile lipsa au fost inlocuite cu media pe coloanele numerice.")
    return data

data = clean_data(data1)

# 2. Statistici descriptive
def descriptive_statistics(data):
    print("\nStatistici descriptive:")
    print(data.describe())

descriptive_statistics(data)

# 3. Vizualizari
# a. Histograme pentru fiecare tip de activitate
def plot_histograms(data):
    plt.figure(figsize=(12, 6))
    data.iloc[:, 1:].hist(bins=10, figsize=(12, 6), color='skyblue', edgecolor='black')
    plt.suptitle("Distributia activitatilor fizice in Europa", fontsize=16)
    plt.show()
    #1. Din histogramă:
#Cele mai multe țări europene prezintă o participare scăzută (sub 15%) la activități combinate de aerobic și masă musculară.
#Activitatea aerobică are o distribuție mai variată, cu unele țări atingând valori ridicate (peste 50%), exemplu fiind țările nordice.
#Participarea la activitățile de masă musculară este mai uniform distribuită, însă în general rămâne mai scăzută decât cea pentru aerobic.

plot_histograms(data)

# b. Barchart pentru fiecare tara
def plot_bar_chart(data):
    data.set_index('Tara')[['Aerobic si masa musculara', 'Aerobic', 'Masa musculara']].plot(
        kind='bar', figsize=(14, 8), color=['#1f77b4', '#ff7f0e', '#2ca02c']
    )
    plt.title("Participarea la activitatile fizice in Europa (Procente)", fontsize=16)
    plt.ylabel("Procentaj", fontsize=14)
    plt.xlabel("Tara", fontsize=14)
    plt.legend(title="Activitate")
    plt.tight_layout()
    plt.show()

plot_bar_chart(data)

# c. Matrice de corelatie intre variabile
#2. Din matricea de corelație:
#Există o corelație puternică între toate cele trei categorii de activități analizate:
#Aerobic și masă musculară: Corelație de 0.96, ceea ce indică faptul că țările cu rate ridicate de participare la una dintre activități au, în general, rate ridicate și la cealaltă.
def plot_correlation_matrix(data):
    plt.figure(figsize=(8, 6))
    sns.heatmap(data.iloc[:, 1:].corr(), annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title("Matricea de corelatie intre activitati fizice", fontsize=16)
    plt.show()

plot_correlation_matrix(data)

# d. Comparare cu media UE din 2020
def compare_with_eu_average(data):
    eu_average = {'Aerobic si masa musculara': 13.6, 'Aerobic': 31.0, 'Masa musculara': 19.7}
    data_means = data.iloc[:, 1:].mean()

    print("\nComparare cu media UE (2020):")
    for activity, value in eu_average.items():
        print(f"Media setului 2019 = {data_means[activity]:.2f} ,{activity}: Media UE din 2020 = {value} ")

compare_with_eu_average(data)
# 4. Clusterizare
# a. Standardizare
#k means este susceptibil la valori mari de aici este necesara standardizarea
def standardize_data(data):
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data.iloc[:, 1:])
    print("Datele au fost standardizate.")
    return data_scaled


data_scaled = standardize_data(data)


# b. Alegerea numarului optim de clustere (metoda Elbow)
#elbow distorsiunea masoara suma patratelor distantelor dintre puncte si centrul clusterului lor
#din punctul x unde scaderea distorsiunii incetineste, acolo este cot ul si numar optim de clustere
def find_optimal_clusters(data_scaled):
    distortions = []
    K = range(1, 10)
    for k in K:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(data_scaled)
        distortions.append(kmeans.inertia_)

    # Grafic pentru Elbow Method
    plt.figure(figsize=(8, 5))
    plt.plot(K, distortions, 'bo-', label="Distorsiune")
    plt.xlabel('Numarul de Clustere')
    plt.ylabel('Distorsiune')
    plt.title('Metoda Elbow pentru numarul optim de clustere')
    plt.legend()
    plt.show()


find_optimal_clusters(data_scaled)

# c. Aplicarea K-Means cu numarul ales de clustere
def apply_kmeans(data_scaled, data, n_clusters=3):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(data_scaled)
    data['Cluster'] = clusters
    print(f"Datele au fost grupate in {n_clusters} clustere.")
    return data


# Aplic K-Means cu 3 clustere
data = apply_kmeans(data_scaled, data, n_clusters=3)


# d. Vizualizare rezultate (clustere pe grafic scatter)
def plot_clusters(data, data_scaled):
    plt.figure(figsize=(10, 6))
    plt.scatter(data_scaled[:, 0], data_scaled[:, 1], c=data['Cluster'], cmap='viridis', s=100)
    plt.title('Clusterizarea Tarilor pe Baza Activitatilor Fizice')
    plt.xlabel('Componenta Principala 1')
    plt.ylabel('Componenta Principala 2')
    plt.colorbar(label='Cluster')
    plt.show()


plot_clusters(data, data_scaled)

# e. Rezultate grupate
print("\nDatele cu clusterele atribuite:")
print(data[['Tara', 'Cluster']])