import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score

CLUSTER_COUNT = 5
COLORMAP = np.random.rand(CLUSTER_COUNT, 3)  # ilosc klastrow oraz 3 - RGB sklad kolorow
ITERATIONS = 5


def prep_data():
    data = pd.read_csv('beacon_readings.csv')
    # print(data.info())
    # print(data)

    # distance
    # odległości robota od trzech radiolatarni
    dist_A = data['Distance A']
    dist_B = data['Distance B']
    dist_C = data['Distance C']

    # positon
    # służy do wyświetlenia wyników
    pos_X = data['Position X'] = data.apply(lambda row: row['Position X'] + random.randint(-6, 6),
                                            axis=1)  # random chmurki punktów
    pos_Y = data['Position Y'] = data.apply(lambda row: row['Position Y'] + random.randint(-6, 6),
                                            axis=1)  # random chmurki punktów
    # plt.scatter(pos_X, pos_Y)
    # plt.xlabel('Positon X')
    # plt.ylabel('Position Y')
    # plt.show()

    return dist_A.to_numpy(), dist_B.to_numpy(), dist_C.to_numpy(), pos_X.to_numpy(), pos_Y.to_numpy(),
    # return data.to_numpy()


def show_cluster(x, y, centroids, assignments, i):
    global COLORMAP
    # przypisanie kolorow
    COLORMAP = np.random.rand(i, 3)  # ilosc klastrow oraz 3 - RGB sklad kolorow
    colors = COLORMAP[assignments]
    # wyswietlenie wykresu
    plt.scatter(x, y, c=colors, alpha=0.2, edgecolors='none')
    # plt.scatter(centroids[:, 0], centroids[:, 1], c=COLORMAP)
    plt.grid(True)
    plt.title('For ' + i.__str__() + ' clusters')
    plt.xlabel('Positon X')
    plt.ylabel('Position Y')
    plt.show()


# pierwsze rozwiazanie ze strema (test dzialania k-means):
def cluster_with_sklearn(a, b, c, x, y):
    global COLORMAP
    for i in range(2, CLUSTER_COUNT + 1):
        kmeans = KMeans(n_clusters=i)
        data = np.array([a, b, c]).T
        kmeans.fit(data)
        show_cluster(x, y, kmeans.cluster_centers_, kmeans.labels_, i)
        db_index = davies_bouldin_score(data, kmeans.labels_)
        print("for " + i.__str__() + " clusters " + round(db_index, 4).__str__())


# WLASNY K-MEANS:
def generate_assignments(a, b, c, centroids):
    distances = []
    # zamienic x i y na a b c
    for centroid in centroids:
        dist = np.sqrt((a - centroid[0]) ** 2 + (b - centroid[1]) ** 2 + (c - centroid[2]) ** 2)  # dystans euqlidesowy
        distances.append(dist)
    distances = np.array(distances)
    assignments = np.argmin(distances, axis=0)
    #print(assignments)
    return assignments


def generate_centroids(data, i):
    centroids = []

    min_distances = data.min(axis=0)
    max_distances = data.max(axis=0)
    median_distance = np.median(max_distances) / 3  # aby minimalna wartosc nie byla skrajnie 0

    # print(median_distance)
    # print(max_distances)
    # print(min_distances)

    for _ in range(i):
        new_centroid = []
        for min_distance, max_distance in zip(min_distances, max_distances):
            new_centroid.append(np.random.uniform(min_distance + median_distance, max_distance))
        centroids.append(new_centroid)
    #print(np.array(centroids))
    return np.array(centroids)


def update_centroids(a, b, c, assignments, old_centroids):
    new_centroids = []
    points = np.array([a, b, c]).T
    for idx, old_centroid in enumerate(old_centroids):
        cluster = points[assignments == idx, :]
        if len(cluster) == 0:
            new_centroids.append(old_centroid)
        else:
            new_centroids.append(np.mean(cluster, axis=0))
    return np.array(new_centroids, dtype=object)


def kmeans_from_scratch(a, b, c, x, y):
    global COLORMAP

    # petla
    for i in range(2, CLUSTER_COUNT + 1):
        # generujemy losowe klastry/centroidy
        centroids = generate_centroids(np.array([a, b, c]).T, i)  # np.random.rand(i, 3) + 0.8   # dobor zakresu
        assignments = generate_assignments(a, b, c, centroids)

        for _ in range(ITERATIONS):
            assignments = generate_assignments(a, b, c, centroids)
            centroids = update_centroids(a, b, c, assignments, centroids)

        show_cluster(x, y, centroids, assignments, i)
        data = np.array([a, b, c]).T
        db_index = davies_bouldin_score(data, assignments)
        print("for " + i.__str__() + " clusters " + round(db_index, 4).__str__())


def main():
    dist_A, dist_B, dist_C, pos_X, pos_Y = prep_data()
    # data = prep_data()

    # test zadania ze streama:
    # cluster_with_sklearn(dist_A, dist_B, dist_C, pos_X, pos_Y)

    # Przeanalizuj stosując algorytm K-średnich zaimplementowany samodzielnie
    # dla ilu klastrów uzyskujemy najlepszy wynik? 2-3-4-5?
    # stosujemy k-means dla DISTANCE
    # wyswietlamy odpowiednie POSITION dla pogrypowanych dystansów
    kmeans_from_scratch(dist_A, dist_B, dist_C, pos_X, pos_Y)


if __name__ == '__main__':
    main()
