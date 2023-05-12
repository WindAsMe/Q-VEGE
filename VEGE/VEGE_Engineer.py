#coding:UTF-8
'''
Created by Jun YU (yujun@ie.niigata-u.ac.jp) on November 22, 2022
benchmark function: 28 functions of the CEC2017 test suite (https://www3.ntu.edu.sg/home/epnsugan/index_files/CEC2017/CEC2017.htm)
reference paper: Jun Yu, "Vegetation Evolution: An Optimization Algorithm Inspired by the Life Cycle of Plants," 
                          International Journal of Computational Intelligence and Applications, vol. 21, no.2, Article No. 2250010
'''
import os

import numpy as np
from Problems.CSD import CSD_obj, CSD_cons
from Problems.PVD import PVD_obj, PVD_cons
from Problems.TBT import TBT_obj, TBT_cons
from Problems.WBD import WBD_obj, WBD_cons


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
current_lifespan = 0                                                  # the current growth cycle of an individual
Current_fitness_evaluations = 0                                       # the current number of fitness evaluations
Fun_num = None

def isNegative(cons):
    for con in cons:
        if con > 0:
            return False
    return True


def Evaluation(indi, obj_func, cons_func):
    global Current_fitness_evaluations
    obj = obj_func(indi)
    cons = cons_func(indi)
    if isNegative(cons) == False:
        obj = FITNESS_MAX
    Current_fitness_evaluations += 1
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
    global Population, Population_fitness, current_lifespan
    for i in range(POPULATION_SIZE):
        for j in range(DIMENSION_NUM):
            Population[i][j] = np.random.uniform(LOWER_BOUNDARY[j], UPPER_BOUNDARY[j])
        Population_fitness[i] = Evaluation(Population[i], obj_func, cons_func)
    current_lifespan = 1


def Growth(obj_func, cons_func):
    global Population, Population_fitness
    offspring = np.zeros((POPULATION_SIZE, DIMENSION_NUM))
    offspring_fitness = np.zeros(POPULATION_SIZE)
    for i in range(POPULATION_SIZE):
        for j in range(DIMENSION_NUM):
            offspring[i][j] = Population[i][j] + GR * (np.random.random() * 2.0 - 1.0)
        CheckIndi(offspring[i])
        offspring_fitness[i] = Evaluation(offspring[i], obj_func, cons_func)
        if offspring_fitness[i] < Population_fitness[i]:
            Population_fitness[i] = offspring_fitness[i]
            Population[i] = offspring[i].copy()


def Maturity(obj_func, cons_func):
    global Population, Population_fitness
    seed_individual = np.zeros((POPULATION_SIZE*SEED_NUM, DIMENSION_NUM))
    seed_individual_fitness = np.zeros(POPULATION_SIZE*SEED_NUM)
    for i in range(POPULATION_SIZE):
        for j in range(SEED_NUM):
            index1 = index2 = 0
            while index1 == i:
                index1 = np.random.randint(0, POPULATION_SIZE)
            while index2 == i or index2 == index1:
                index2 = np.random.randint(0, POPULATION_SIZE)
            seed_individual[i*SEED_NUM + j] = Population[i] + MS * (np.random.random() * 2.0 - 1.0) * (Population[index1] - Population[index2])
            CheckIndi(seed_individual[i*SEED_NUM + j])
            seed_individual_fitness[i*SEED_NUM + j] = Evaluation(seed_individual[i*SEED_NUM + j], obj_func, cons_func)
    # Select the top PS individuals from the current population and seeds to enter the next generation
    temp_individual = np.vstack((Population, seed_individual))
    temp_individual_fitness = np.hstack((Population_fitness, seed_individual_fitness))
    tmp = list(map(list, zip(range(len(temp_individual_fitness)), temp_individual_fitness)))
    small = sorted(tmp, key=lambda x: x[1], reverse=False)
    for i in range(POPULATION_SIZE):
        key, _ = small[i]
        Population_fitness[i] = temp_individual_fitness[key]
        Population[i] = temp_individual[key].copy()


# the implementation process of differential evolution
def VegetationEvolution(obj_func, cons_func):
    global current_lifespan, GC
    if current_lifespan < GC:
        Growth(obj_func, cons_func)
        current_lifespan += 1
    elif current_lifespan == GC:
        Maturity(obj_func, cons_func)
        current_lifespan = 0
    else:
        print("Error: Maximum generation period exceeded.")


def RunVEGE(obj_func, cons_func):
    global Current_fitness_evaluations, Fun_num, Population_fitness
    All_Trial_Best = []
    for i in range(REPETITION_NUM):
        Best_list = []
        Current_fitness_evaluations = 0
        Current_generation = 1
        np.random.seed(2022 + 88*i)                 # fix the seed of random number
        Initialization(obj_func, cons_func)                            # randomly initialize the population
        Best_list.append(min(Population_fitness))
        while Current_fitness_evaluations < MAX_FITNESS_EVALUATION_NUM:
            VegetationEvolution(obj_func, cons_func)
            Current_generation = Current_generation + 1
            Best_list.append(min(Population_fitness))
        All_Trial_Best.append(Best_list)
    np.savetxt('./VEGE_Data/Engineer/' + Fun_num + '.csv', All_Trial_Best, delimiter=",")


def main():
    global Fun_num, DIMENSION_NUM, Population, MAX_FITNESS_EVALUATION_NUM, LOWER_BOUNDARY, UPPER_BOUNDARY

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
    for i in range(len(Dims)-1):
        DIMENSION_NUM = Dims[i]
        LOWER_BOUNDARY = scale_range[i][0]
        UPPER_BOUNDARY = scale_range[i][1]
        Population = np.zeros((POPULATION_SIZE, DIMENSION_NUM))
        Fun_num = problem_names[i]
        RunVEGE(obj_funcs[i], cons_funcs[i])


if __name__ == "__main__":
    if os.path.exists('./VEGE_Data/Engineer') == False:
        os.makedirs('./VEGE_Data/Engineer')
    main()