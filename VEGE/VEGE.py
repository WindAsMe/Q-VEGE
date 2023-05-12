#coding:UTF-8
'''
Created by Jun YU (yujun@ie.niigata-u.ac.jp) on November 22, 2022
benchmark function: 28 functions of the CEC2017 test suite (https://www3.ntu.edu.sg/home/epnsugan/index_files/CEC2017/CEC2017.htm)
reference paper: Jun Yu, "Vegetation Evolution: An Optimization Algorithm Inspired by the Life Cycle of Plants," 
                          International Journal of Computational Intelligence and Applications, vol. 21, no.2, Article No. 2250010
'''
import os

import numpy as np
from opfunu.cec_based import cec2013, cec2014


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
SuiteName = "CEC2013"


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
def Initialization(func):
    global Population, Population_fitness, CFEs, current_lifespan
    for i in range(POPULATION_SIZE):
        for j in range(DIMENSION_NUM):
            Population[i][j] = np.random.uniform(LOWER_BOUNDARY[j], UPPER_BOUNDARY[j])
        Population_fitness[i] = func.evaluate(Population[i])
        CFEs += 1
    current_lifespan = 1


def Growth(func):
    global Population, Population_fitness, CFEs
    offspring = np.zeros((POPULATION_SIZE, DIMENSION_NUM))
    offspring_fitness = np.zeros(POPULATION_SIZE)
    for i in range(POPULATION_SIZE):
        for j in range(DIMENSION_NUM):
            offspring[i][j] = Population[i][j] + GR * (np.random.random() * 2.0 - 1.0)
        CheckIndi(offspring[i])
        offspring_fitness[i] = func.evaluate(offspring[i])
        CFEs += 1
        if offspring_fitness[i] < Population_fitness[i]:
            Population_fitness[i] = offspring_fitness[i]
            Population[i] = offspring[i].copy()


def Maturity(func):
    global Population, Population_fitness, CFEs
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
            seed_individual_fitness[i*SEED_NUM + j] = func.evaluate(seed_individual[i*SEED_NUM + j])
            CFEs += 1
    # Select the top PS individuals from the current population and seeds to enter the next generation
    temp_individual = np.vstack((Population, seed_individual))
    temp_individual_fitness = np.hstack((Population_fitness, seed_individual_fitness))
    tmp = list(map(list, zip(range(len(temp_individual_fitness)), temp_individual_fitness)))
    small = sorted(tmp, key=lambda x: x[1], reverse=False)
    for i in range(POPULATION_SIZE):
        key, _ = small[i]
        Population_fitness[i] = temp_individual_fitness[key]
        Population[i] = temp_individual[key].copy()


def VegetationEvolution(bench):
    global current_lifespan, GC, CFEs
    if current_lifespan < GC:
        Growth(bench)
        current_lifespan += 1
    elif current_lifespan == GC:
        Maturity(bench)
        current_lifespan = 0
    else:
        print("Error: Maximum generation period exceeded.")


def RunVEGE(func):
    global CFEs, Fun_num, Population_fitness, SuiteName
    All_Trial_Best = []
    for i in range(REPETITION_NUM):
        Best_list = []
        CFEs = 0
        np.random.seed(2022 + 88 * i)
        Initialization(func)
        Best_list.append(min(Population_fitness))
        while CFEs < MAX_FITNESS_EVALUATION_NUM:
            VegetationEvolution(func)
            Best_list.append(min(Population_fitness))
        All_Trial_Best.append(Best_list)
    with open('./VEGE_Data/' + SuiteName + '/F{}_{}D.csv'.format(Fun_num, DIMENSION_NUM), "ab") as f:
        np.savetxt(f, All_Trial_Best, delimiter=",")


def main(Dim):
    global Fun_num, DIMENSION_NUM, Population, MAX_FITNESS_EVALUATION_NUM, SuiteName, LOWER_BOUNDARY, UPPER_BOUNDARY
    DIMENSION_NUM = Dim
    Population = np.zeros((POPULATION_SIZE, DIMENSION_NUM))
    MAX_FITNESS_EVALUATION_NUM = DIMENSION_NUM * 500
    LOWER_BOUNDARY = [-100] * DIMENSION_NUM
    UPPER_BOUNDARY = [100] * DIMENSION_NUM

    CEC2013Funcs = [cec2013.F12013(Dim), cec2013.F22013(Dim), cec2013.F32013(Dim), cec2013.F42013(Dim),
                    cec2013.F52013(Dim), cec2013.F62013(Dim), cec2013.F72013(Dim), cec2013.F82013(Dim),
                    cec2013.F92013(Dim), cec2013.F102013(Dim), cec2013.F112013(Dim), cec2013.F122013(Dim),
                    cec2013.F132013(Dim), cec2013.F142013(Dim), cec2013.F152013(Dim), cec2013.F162013(Dim),
                    cec2013.F172013(Dim), cec2013.F182013(Dim), cec2013.F192013(Dim), cec2013.F202013(Dim),
                    cec2013.F212013(Dim), cec2013.F222013(Dim), cec2013.F232013(Dim), cec2013.F242013(Dim),
                    cec2013.F252013(Dim), cec2013.F262013(Dim), cec2013.F272013(Dim), cec2013.F282013(Dim)]

    CEC2014Funcs = [cec2014.F12014(Dim), cec2014.F22014(Dim), cec2014.F32014(Dim), cec2014.F42014(Dim),
                    cec2014.F52014(Dim), cec2014.F62014(Dim), cec2014.F72014(Dim), cec2014.F82014(Dim),
                    cec2014.F92014(Dim), cec2014.F102014(Dim), cec2014.F112014(Dim), cec2014.F122014(Dim),
                    cec2014.F132014(Dim), cec2014.F142014(Dim), cec2014.F152014(Dim), cec2014.F162014(Dim),
                    cec2014.F172014(Dim), cec2014.F182014(Dim), cec2014.F192014(Dim), cec2014.F202014(Dim),
                    cec2014.F212014(Dim), cec2014.F222014(Dim), cec2014.F232014(Dim), cec2014.F242014(Dim),
                    cec2014.F252014(Dim), cec2014.F262014(Dim), cec2014.F272014(Dim), cec2014.F282014(Dim),
                    cec2014.F292014(Dim), cec2014.F302014(Dim)]
    # CEC_dict = {"CEC2013": CEC2013Funcs, "CEC2014": CEC2014Funcs}
    # for suite in CEC_dict.keys():
    #     Fun_num = 0
    #     for i in range(len(CEC_dict[SuiteName])):
    #         SuiteName = suite
    #         Fun_num = i+1
    #         RunVEGE(CEC_dict[SuiteName][i])
    Fun_num = 0
    for i in range(28, len(CEC2014Funcs)):
        SuiteName = "CEC2014"
        Fun_num = i + 1
        RunVEGE(CEC2014Funcs[i])


if __name__ == "__main__":
    if os.path.exists('./VEGE_Data/CEC2013') == False:
        os.makedirs('./VEGE_Data/CEC2013')
    if os.path.exists('./VEGE_Data/CEC2014') == False:
        os.makedirs('./VEGE_Data/CEC2014')
    Dims = [10]
    for Dim in Dims:
        main(Dim)
