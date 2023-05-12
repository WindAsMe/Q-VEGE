from mealpy.system_based import AEO
import os
from opfunu.cec_based import cec2013, cec2014
import numpy as np
import warnings
warnings.filterwarnings("ignore")


def main(Dim):
    trials = 30
    PopSize = 100
    MaxFEs = 500 * Dim
    MaxIter = int(MaxFEs / PopSize)
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
    CEC_dict = {"CEC2013": CEC2013Funcs, "CEC2014": CEC2014Funcs}
    for suite in CEC_dict.keys():
        SuiteName = suite
        for i in range(len(CEC_dict[SuiteName])):
            problem_dict = {
                "fit_func": CEC_dict[SuiteName][i].evaluate,
                "lb": [-100] * Dim,
                "ub": [100] * Dim,
                "minmax": "min",
                "log_to": None
            }
            All_Trial_Best = []
            for j in range(trials):
                aeo_solver = AEO.OriginalAEO(epoch=MaxIter, pop_size=PopSize)
                np.random.seed(2022 + 88 * j)
                best_position, best_fitness_value = aeo_solver.solve(problem_dict)
                All_Trial_Best.append(aeo_solver.history.list_global_best_fit)
            np.savetxt("./AEO_Data/" + SuiteName + "/F" + str(i+1) + "_" + str(Dim) + "D.csv", All_Trial_Best, delimiter=",")


if __name__ == "__main__":
    if os.path.exists('./AEO_Data/CEC2013') == False:
        os.makedirs('./AEO_Data/CEC2013')
    if os.path.exists('./AEO_Data/CEC2014') == False:
        os.makedirs('./AEO_Data/CEC2014')
    Dims = [10, 30, 50]
    for Dim in Dims:
        main(Dim)