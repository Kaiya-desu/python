# algorytm uczenia maszynowego, gra
# program gra z nami w kamien papier nozyce,  p->k->n->p (...)
# za pomocą łańcucha markowa
# co jest stanem? -> wygranie/przegranie
# algorytm na początku wybiera co zagra , nastepnie gracz wybiera i algorytm zapisuje wynik
# po zagraniu algorytm patrzy jaka jest sytuacja i jak do niej doszło na podstawie tego zmienia wage
# robimy macierz, kolumny odpowiadają źródłowemu stanowi -> P , K , N a wiersze drugiej sytuacji
# mnozmymy przez wektor prawdopodobieństw,

import argparse
from pydoc import doc
import numpy as np
import os


def divideMatrix(matrix, number):
    p = []
    for i in range(len(matrix)):
        p.append(matrix[i] / number)
    return p


def myFunction():
    #     P   K   N  prawdopodobieństwo że wystąpi
    # P   1   1   1
    # K   1   1   1
    # N   1   1   1

    start = ['P', 'K', 'N']

    # DODATKOWE PUNKTY: Istnieje możliwość uzyskania 2 ekstra punktów za realizację zapisu i odczytu modelu z pliku.
    if os.stat('matrixData.txt').st_size == 0:
        matrix = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
    else:
        matrix = np.loadtxt('matrixData.txt')
        print("Ładowanie macierzy z pliku")
        print(matrix)

    countMatrix = np.sum(matrix, axis=1)  # wygodniejsze niz cumsum
    print(countMatrix)

    movesList = ['P']

    # 1. Użytkownik uruchamia skrypt podając jako parametr liczbę punktów dla których ma się gra zakończyć
    def parse_arguments():
        parser = argparse.ArgumentParser(description=doc)
        parser.add_argument('-pkt', '--LiczbaPkt', type=int, required=True, help='Liczba punktów')
        return parser.parse_args()

    args = parse_arguments()
    print(f'Liczba pkt : {args.LiczbaPkt}')

    pointsLimit = args.LiczbaPkt

    points = 0
    rounds = 0

    # 6. Sprawdzane jest czy uzyskano docelową ilość punktów (wartość ujemna oznacza wygraną komputera)
    while abs(points) < pointsLimit:
        rounds = rounds + 1
        print("Runda: " + str(rounds))

        # 3. Użytkownik podaje swoje zagranie
        used = input()
        while used != 'P':
            if used == 'K' or used == 'N':
                break
            print("Zły symbol, podaj [P, K, N]")
            used = input()

        # 2. Komputer wybiera w sekrecie swoje zagranie
        if movesList[len(movesList) - 1] == 'P':
            l = divideMatrix(matrix[0], countMatrix[0])
            state = np.random.choice(start, replace=True, p=l)
        if movesList[len(movesList) - 1] == 'K':
            l = divideMatrix(matrix[1], countMatrix[1])
            state = np.random.choice(start, replace=True, p=l)
        if movesList[len(movesList) - 1] == 'N':
            l = divideMatrix(matrix[2], countMatrix[2])
            state = np.random.choice(start, replace=True, p=l)

        # zamieniamy na kontre
        if state == 'N':
            state = 'K'
        elif state == 'K':
            state = 'P'
        else:
            state = 'N'

        # 4. Wypisywany jest rezultat (np. K-N)
        print('Komputer: ' + state + ' | Gracz: ' + used)
        if (used == 'P' and state == 'N') or (used == 'K' and state == 'P') or (used == 'N' and state == 'K'):
            points -= 1
            print('Komputer wygrał')
        elif (used == 'P' and state == 'K') or (used == 'K' and state == 'N') or (used == 'N' and state == 'P'):
            points += 1
            print('Gracz wygral')
        else:
            print("Remis")

        # 7. Uczenie maszynowe aktualizuje łańcuch Markowa w oparciu o ostatnią turę
        if movesList[len(movesList) - 1] == 'P':
            countMatrix[0] += 1
            if used == 'P':
                matrix[0][0] += 1
            if used == 'K':
                matrix[0][1] += 1
            if used == 'N':
                matrix[0][2] += 1

        if movesList[len(movesList) - 1] == 'K':
            countMatrix[1] += 1
            if used == 'P':
                matrix[1][0] += 1
            if used == 'K':
                matrix[1][1] += 1
            if used == 'N':
                matrix[1][2] += 1

        if movesList[len(movesList) - 1] == 'N':
            countMatrix[2] += 1
            if used == 'P':
                matrix[2][0] += 1
            if used == 'K':
                matrix[2][1] += 1
            if used == 'N':
                matrix[2][2] += 1

        movesList.append(used)
        print("Punkty: " + str(points))
        print("\n")

    print(matrix)

    # DODATKOWE PUNKTY: Istnieje możliwość uzyskania 2 ekstra punktów za realizację zapisu i odczytu modelu z pliku.
    mat = np.matrix(matrix)
    with open('matrixData.txt', 'wb') as f:
        for line in mat:
            np.savetxt(f, line, fmt='%.0f')

    # dodac if __name__ ; divide matrix zastapic macierz/macierz.cumsum()


if __name__ == "__main__":
    myFunction()
