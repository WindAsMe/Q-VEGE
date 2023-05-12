import numpy as np

def Sphere(X):
    Dim = len(X)
    result = 0
    for i in range(Dim):
        result += X[i] ** 2
    return result


def Schwefel(X):
    Dim = len(X)
    part1 = 0
    part2 = 1
    for i in range(Dim):
        part1 += abs(X[i])
        part2 *= abs(X[i])
    return part1 + part2


def Rosenbrock(X):
    Dim = len(X)
    result = 0
    for i in range(Dim-1):
        result += 100 * (X[i] ** 2 - X[i+1]) ** 2 + (X[i] - 1) ** 2
    return result


def Rastrigin(X):
    Dim = len(X)
    result = 0
    for i in range(Dim):
        result += X[i] ** 2 - 10 * np.cos(2 * np.pi * X[i]) + 10
    return result


def Griewank(X):
    Dim = len(X)
    part1 = 0
    part2 = 1
    for i in range(Dim):
        part1 += X[i] ** 2
        part2 *= np.cos(X[i] / np.sqrt(i+1))
    return 1 / 4000 * part1 - part2 + 1


def Ackley(X):
    Dim = len(X)
    part1 = 0
    part2 = 0
    for i in range(Dim):
        part1 += X[i] ** 2
        part2 += np.cos(2 * np.pi * X[i])
    part1 = -0.2 * np.sqrt(1 / Dim * part1)
    part2 = 1 / Dim * part2
    return -20 * np.exp(part1) - np.exp(part2) + 20 + np.e



