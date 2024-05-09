from pulp import *
import matplotlib.pyplot as plt
import itertools

### macierz odleglosci
D = [
    [16.160, 24.080, 24.320, 21.120],
    [19.000, 26.470, 27.240, 17.330],
    [25.290, 32.490, 33.420, 12.250],
    [0.000, 7.930, 8.310, 36.120],
    [3.070, 6.440, 7.560, 37.360],
    [1.220, 7.510, 8.190, 36.290],
    [2.800, 10.310, 10.950, 33.500],
    [2.870, 5.070, 5.670, 38.800],
    [3.800, 8.010, 7.410, 38.160],
    [12.350, 4.520, 4.350, 48.270],
    [11.110, 3.480, 2.970, 47.140],
    [21.990, 22.020, 24.070, 39.860],
    [8.820, 3.300, 5.360, 43.310],
    [7.930, 0.000, 2.070, 43.750],
    [9.340, 2.250, 1.110, 45.430],
    [8.310, 2.070, 0.000, 44.430],
    [7.310, 2.440, 1.110, 43.430],
    [7.550, 0.750, 1.530, 43.520],
    [11.130, 18.410, 19.260, 25.400],
    [17.490, 23.440, 24.760, 23.210],
    [11.030, 18.930, 19.280, 25.430],
    [36.120, 43.750, 44.430, 0.000]
]

### pracochlonnosc
P = [0.1609, 0.1164, 0.1026, 0.1516, 0.0939, 0.1320, 0.0687, 0.0930, 0.2116, 0.2529, 0.0868, 0.0828, 0.0975, 0.8177,
     0.4115, 0.3795, 0.0710, 0.0427, 0.1043, 0.0997, 0.1698, 0.2531]

### OBECNY PRZYDZIAL
A = [
    [0, 0, 0, 1],
    [0, 0, 0, 1],
    [0, 0, 0, 1],
    [1, 0, 0, 0],
    [1, 0, 0, 0],
    [1, 0, 0, 0],
    [1, 0, 0, 0],
    [1, 0, 0, 0],
    [0, 0, 1, 0],
    [0, 1, 0, 0],
    [0, 1, 0, 0],
    [0, 1, 0, 0],
    [0, 1, 0, 0],
    [0, 1, 0, 0],
    [1, 0, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 1, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1],
    [0, 0, 0, 1],
    [0, 0, 0, 1],
    [0, 0, 0, 1],
]

### OBECNE LOKALIZACJE SIEDZIB
L = [4, 14, 16, 22]

epsilons = [i for i in range(0,101,1)] #sprawdzany zakres epsilon dla f2
solutions = set()
# Inicjalizacja modelu
for epsilon in epsilons:
    model = LpProblem("Pfitzer Problem", LpMinimize)

    # Definicja zmiennych decyzyjnych
    x = LpVariable.dicts('x', [(i, j) for i in range(len(A)) for j in range(len(A[0]))], cat=LpBinary)

    # f1
    model += lpSum(D[i][j] * x[(i, j)] for i in range(len(A)) for j in range(len(A[0])))

    # f2 - ograniczone przez epsilon
    model += lpSum(x[(i, j)] * (1 - A[i][j]) * P[i] / sum(P) for i in range(len(A)) for j in range(len(A[0]))) <= epsilon


    # Ograniczenia
    for i in range(len(A)):
        model += lpSum(x[(i, j)] for j in range(len(A[0]))) == 1  # Każdy region ma być przypisany dokładnie jednemu przedstawicielowi

    for j in range(len(A[0])):
        model += lpSum(x[(i, j)] * P[i] for i in range(len(A))) >= 0.9  # Suma pracochłonności przydzielonych regionów >= 0.9
        model += lpSum(x[(i, j)] * P[i] for i in range(len(A))) <= 1.1  # Suma pracochłonności przydzielonych regionów <= 1.1
    
    model.solve()

    solutions.add((value(model.objective),sum(x[(i, j)].value() * (1 - A[i][j]) * P[i] / sum(P) for i in range(len(A)) for j in range(len(A[0])))))

epsilons = [i for i in range(0,201,7)] #sprawdzany zakres epsilon dla f1
for epsilon in epsilons:
    model = LpProblem("Pfitzer Problem", LpMinimize)

    # Definicja zmiennych decyzyjnych
    x = LpVariable.dicts('x', [(i, j) for i in range(len(A)) for j in range(len(A[0]))], cat=LpBinary)

    # f1 - ograniczone przez epsilon
    model += lpSum(D[i][j] * x[(i, j)] for i in range(len(A)) for j in range(len(A[0]))) <= epsilon

    # f2 
    model += lpSum(x[(i, j)] * (1 - A[i][j]) * P[i] / sum(P) for i in range(len(A)) for j in range(len(A[0])))

    # Ograniczenia
    for i in range(len(A)):
        model += lpSum(x[(i, j)] for j in range(len(A[0]))) == 1  # Każdy region ma być przypisany dokładnie jednemu przedstawicielowi

    for j in range(len(A[0])):
        model += lpSum(x[(i, j)] * P[i] for i in range(len(A))) >= 0.9  # Suma pracochłonności przydzielonych regionów >= 0.9
        model += lpSum(x[(i, j)] * P[i] for i in range(len(A))) <= 1.1  # Suma pracochłonności przydzielonych regionów <= 1.1
    
    model.solve()

    solutions.add((sum(D[i][j] * x[(i, j)].value() for i in range(len(A)) for j in range(len(A[0]))),value(model.objective)))

#usun zdominowane rozwiazania
found = True
while found:
    found = False
    for i, val in enumerate(itertools.islice(solutions, len(solutions))):
        for ii, val2 in enumerate(itertools.islice(solutions, len(solutions))):
            if i == ii:
                continue
            if val[0] > val2[0] and val[1] > val2[1]:
                found = True
                solutions.remove(val)
                break
        if found:
            break


#wyswietl 10 rozwiazan
xpoints = []
ypoints = []
plt.plot()
for i, val in enumerate(itertools.islice(solutions, 10)):
    xpoints.append(val[0])
    ypoints.append(val[1])

plt.plot(xpoints, ypoints, 'o')
plt.show()