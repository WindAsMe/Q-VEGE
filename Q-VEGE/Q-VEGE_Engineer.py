#coding:UTF-8
'''
Created by Jun YU (yujun@ie.niigata-u.ac.jp) on November 22, 2022
benchmark function: 28 functions of the CEC2017 test suite (https://www3.ntu.edu.sg/home/epnsugan/index_files/CEC2017/CEC2017.htm)
reference paper: Jun Yu, "Vegetation Evolution: An Optimization Algorithm Inspired by the Life Cycle of Plants," 
                          International Journal of Computational Intelligence and Applications, vol. 21, no.2, Article No. 2250010
'''
import os
from copy import deepcopy
import numpy as np
from Problems.CSD import CSD_obj, CSD_cons
from Problems.PVD import PVD_obj, PVD_cons
from Problems.TBT import TBT_obj, TBT_cons
from Problems.WBD import WBD_obj, WBD_cons
from scipy.stats import levy
from pyDOE2 import lhs

FITNESS_MAX = 1e50
POPULATION_SIZE = 10                                                  # the number of individuals (POPULATION_SIZE > 4)
DIMENSION_NUM = 10                                                    # the number of variables
LOWER_BOUNDARY = [-100] * DIMENSION_NUM
UPPER_BOUNDARY = [100] * DIMENSION_NUM                                                 # the minimum value of the variable range
REPETITION_NUM = 30                                                   # the number of independent runs
MAX_FITNESS_EVALUATION_NUM = DIMENSION_NUM * 500                      # the maximum number of fitness evaluations
GC = 6                                                                # the maximum growth cycle of an individual
GR = 1                                                                # the maximum growth radius of an individual
MS = 2                                                                # the moving scale
SEED_NUM = 6                                                          # the number of generated seeds by each individual

Population = np.zeros((POPULATION_SIZE, DIMENSION_NUM))               # the coordinates of the individual (candidate solutions)
Population_fitness = np.zeros(POPULATION_SIZE)                        # the fitness value of all individuals
current_lifespan = 0
CFEs = 0                                                              # the current number of fitness evaluations
Fun_num = None
SuiteName = "Engineer"
Q_table_exploration = np.zeros(4)
Q_table_exploitation = np.zeros(4)
epsilon = 0.5
Gamma = 1


def isNegative(cons):
    for con in cons:
        if con > 0:
            return False
    return True


def Evaluation(indi, obj_func, cons_func):
    global CFEs
    obj = obj_func(indi)
    cons = cons_func(indi)
    if isNegative(cons) == False:
        obj = FITNESS_MAX
    CFEs += 1
    return obj


def CheckIndi(Indi):
    for i in range(DIMENSION_NUM):
        range_width = UPPER_BOUNDARY[i] - LOWER_BOUNDARY[i]
        if Indi[i] > UPPER_BOUNDARY[i]:
            n = int((Indi[i] - UPPER_BOUNDARY[i]) / range_width)
            mirrorRange = (Indi[i] - UPPER_BOUNDARY[i]) - (n * range_width)
            Indi[i] = UPPER_BOUNDARY[i] - mirrorRange
        elif Indi[i] < LOWER_BOUNDARY[i]:
            n = int((LOWER_BOUNDARY[i] - Indi[i]) / range_width)
            mirrorRange = (LOWER_BOUNDARY[i] - Indi[i]) - (n * range_width)
            Indi[i] = LOWER_BOUNDARY[i] + mirrorRange
        else:
            pass


# initialize the population randomly
def Initialization(obj_func, cons_func):
    global Population, Population_fitness, CFEs, current_lifespan
    Population = lhs(DIMENSION_NUM, samples=POPULATION_SIZE)
    for i in range(POPULATION_SIZE):
        for j in range(DIMENSION_NUM):
            Population[i][j] = LOWER_BOUNDARY[j] + (UPPER_BOUNDARY[j] - LOWER_BOUNDARY[j]) * Population[i][j]
        Population_fitness[i] = Evaluation(Population[i], obj_func, cons_func)
        CFEs += 1
    current_lifespan = 1


def LocalSearch(indi):
    global GR, DIMENSION_NUM
    off = deepcopy(indi)
    for i in range(DIMENSION_NUM):
        off[i] += GR * (np.random.random() * 2.0 - 1.0)
    return off


def NormalSearch(indi):
    global GR, DIMENSION_NUM
    off = deepcopy(indi)
    for i in range(DIMENSION_NUM):
        off[i] += GR * (np.random.normal(0, 1))
    return off


def LevySearch(indi):
    global DIMENSION_NUM
    off = deepcopy(indi)
    for i in range(DIMENSION_NUM):
        off[i] += levy.rvs()
    return off


def ChebyshevMap():
    global GR, DIMENSION_NUM
    v = np.zeros(DIMENSION_NUM)
    v[0] = np.random.rand()
    for i in range(1, DIMENSION_NUM):
        v[i] = np.cos(i / np.cos(v[i-1]))
    return GR * v


def ChaosSearch(indi):
    global GR, DIMENSION_NUM
    off = deepcopy(indi)
    off += ChebyshevMap()
    return off


def isZero(Q_table):
    for i in Q_table:
        if i != 0:
            return False
    return True


def epsilonGreedy(Q_table):
    global epsilon
    size = len(Q_table)
    if isZero(Q_table):
        return np.random.randint(0, size)
    else:
        if np.random.rand() < epsilon:
            return np.argmax(Q_table)
        else:
            return np.random.randint(0, size)


def Exploitation(i):
    global Population, Q_table_exploitation
    archive = [LocalSearch, NormalSearch, LevySearch, ChaosSearch]
    index = epsilonGreedy(Q_table_exploitation)
    strategy = archive[index]
    return strategy(Population[i]), index


def Growth(obj_func, cons_func):
    global Population, Population_fitness, CFEs, Q_table_exploitation, Gamma
    Temp_table = np.zeros(4)
    Times_table = [0.00000000001] * 4
    offspring = np.zeros((POPULATION_SIZE, DIMENSION_NUM))
    offspring_fitness = np.zeros(POPULATION_SIZE)
    for i in range(POPULATION_SIZE):
        offspring[i], index = Exploitation(i)
        CheckIndi(offspring[i])
        offspring_fitness[i] = Evaluation(offspring[i], obj_func, cons_func)
        Times_table[index] += 1
        Temp_table[index] += Population_fitness[i] - offspring_fitness[i]
        CFEs += 1
        if offspring_fitness[i] < Population_fitness[i]:
            Population_fitness[i] = offspring_fitness[i]
            Population[i] = offspring[i].copy()
    for i in range(len(Temp_table)):
        Temp_table[i] /= Times_table[i]
    Temp_table *= Gamma
    for i in range(len(Temp_table)):
        Q_table_exploitation[i] += Temp_table[i]


def Cur(i):
    global Population, MS, POPULATION_SIZE
    candi = list(range(0, POPULATION_SIZE))
    candi.remove(i)
    r1, r2 = np.random.choice(candi, 2, replace=False)
    return Population[i] + MS * (np.random.random() * 2.0 - 1.0) * (Population[r1] - Population[r2])


def CurToRand(i):
    global Population, MS, POPULATION_SIZE
    candi = list(range(0, POPULATION_SIZE))
    candi.remove(i)
    r1, r2, r3 = np.random.choice(candi, 3, replace=False)
    return Population[i] + MS * (np.random.random() * 2.0 - 1.0) * (Population[r1] - Population[i]) + MS * (np.random.random() * 2.0 - 1.0) * (Population[r2] - Population[r3])


def CurToBest(i):
    global Population, Population_fitness, MS, POPULATION_SIZE
    X_best = Population[np.argmin(Population_fitness)]
    candi = list(range(0, POPULATION_SIZE))
    candi.remove(i)
    r1, r2 = np.random.choice(candi, 2, replace=False)
    return Population[i] + MS * (np.random.random() * 2.0 - 1.0) * (X_best - Population[i]) + MS * (np.random.random() * 2.0 - 1.0) * (Population[r1] - Population[r2])


def CurTopBest(i):
    global Population, Population_fitness, MS, POPULATION_SIZE, DIMENSION_NUM
    p = 0.5
    size = int(p * POPULATION_SIZE)
    index = np.argsort(Population_fitness)[0:size]
    X_pbest = np.zeros(DIMENSION_NUM)
    for j in index:
        X_pbest += Population[j]
    X_pbest /= size
    candi = list(range(0, POPULATION_SIZE))
    candi.remove(i)
    r1, r2 = np.random.choice(candi, 2, replace=False)
    return Population[i] + MS * (np.random.random() * 2.0 - 1.0) * (X_pbest - Population[i]) + MS * (np.random.random() * 2.0 - 1.0) * (Population[r1] - Population[r2])


def Exploration(i):
    global Population, Q_table_exploration
    archive = [Cur, CurToRand, CurToBest, CurTopBest]
    index = epsilonGreedy(Q_table_exploration)
    strategy = archive[index]
    return strategy(i), index


def Maturity(obj_func, cons_func):
    global Population, Population_fitness, CFEs, Q_table_exploration
    seed_individual = np.zeros((POPULATION_SIZE*SEED_NUM, DIMENSION_NUM))
    seed_individual_fitness = np.zeros(POPULATION_SIZE*SEED_NUM)
    Temp_table = np.zeros(4)
    Times_table = [0.00000000001] * 4
    for i in range(POPULATION_SIZE):
        for j in range(SEED_NUM):
            seed_individual[i*SEED_NUM + j], index = Exploration(i)
            CheckIndi(seed_individual[i*SEED_NUM + j])
            seed_individual_fitness[i*SEED_NUM + j] = Evaluation(seed_individual[i*SEED_NUM + j], obj_func, cons_func)
            Times_table[index] += 1
            Temp_table[index] += seed_individual_fitness[i*SEED_NUM + j] - Population_fitness[i]
            CFEs += 1
    for i in range(len(Temp_table)):
        Temp_table[i] /= Times_table[i]
    Temp_table *= Gamma
    for i in range(len(Temp_table)):
        Q_table_exploration[i] += Temp_table[i]

    temp_individual = np.vstack((Population, seed_individual))
    temp_individual_fitness = np.hstack((Population_fitness, seed_individual_fitness))
    tmp = list(map(list, zip(range(len(temp_individual_fitness)), temp_individual_fitness)))
    small = sorted(tmp, key=lambda x: x[1], reverse=False)
    for i in range(POPULATION_SIZE):
        key, _ = small[i]
        Population_fitness[i] = temp_individual_fitness[key]
        Population[i] = temp_individual[key].copy()


def VegetationEvolution(obj_func, cons_func):
    global current_lifespan, GC, CFEs
    if current_lifespan < GC:
        Growth(obj_func, cons_func)
        current_lifespan += 1
    elif current_lifespan == GC:
        Maturity(obj_func, cons_func)
        current_lifespan = 0
    else:
        print("Error: Maximum generation period exceeded.")


def RunQVEGE(obj_func, cons_func):
    global CFEs, Fun_num, Population_fitness, SuiteName, Gamma, Q_table_exploration, Q_table_exploitation
    All_Trial_Best = []
    for i in range(REPETITION_NUM):
        Best_list = []
        CFEs = 0
        Gamma = 1
        Q_table_exploration = np.zeros(4)
        Q_table_exploitation = np.zeros(4)
        np.random.seed(2022 + 88 * i)
        Initialization(obj_func, cons_func)
        Best_list.append(min(Population_fitness))
        while CFEs < MAX_FITNESS_EVALUATION_NUM:
            VegetationEvolution(obj_func, cons_func)
            Best_list.append(min(Population_fitness))
            if CFEs % 120 == 0:
                Gamma *= 0.9
                # print(Q_table_exploitation)
                # print(Q_table_exploration)
        All_Trial_Best.append(Best_list)
    np.savetxt('./Q_VEGE_Data/Engineer/' + Fun_num + '.csv', All_Trial_Best, delimiter=",")


def main():
    global Fun_num, DIMENSION_NUM, Population, MAX_FITNESS_EVALUATION_NUM, SuiteName, LOWER_BOUNDARY, UPPER_BOUNDARY
    Dims = [3, 4, 2, 4]
    problem_names = ["CSD", "PVD", "TBT", "WBD"]
    obj_funcs = [CSD_obj, PVD_obj, TBT_obj, WBD_obj]
    cons_funcs = [CSD_cons, PVD_cons, TBT_cons, WBD_cons]

    CSD_range = [[0.05, 0.25, 2], [2, 1.3, 15]]
    PVD_range = [[0, 0, 10, 10], [99, 99, 200, 200]]
    TBT_range = [[0, 0], [1, 1]]
    WBD_range = [[0.1, 0.1, 0.1, 0.1], [2, 10, 10, 2]]
    scale_range = [CSD_range, PVD_range, TBT_range, WBD_range]
    MAX_FITNESS_EVALUATION_NUM = 20000
    for i in range(len(Dims) - 1):
        DIMENSION_NUM = Dims[i]
        LOWER_BOUNDARY = scale_range[i][0]
        UPPER_BOUNDARY = scale_range[i][1]
        Population = np.zeros((POPULATION_SIZE, DIMENSION_NUM))
        Fun_num = problem_names[i]
        RunQVEGE(obj_funcs[i], cons_funcs[i])



if __name__ == "__main__":
    if os.path.exists('./Q_VEGE_Data/Engineer') == False:
        os.makedirs('./Q_VEGE_Data/Engineer')
    main()
