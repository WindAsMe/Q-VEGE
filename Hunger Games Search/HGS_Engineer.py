from mealpy.swarm_based import HGS
import os
from Problems.CSD import CSD_obj, CSD_cons
from Problems.PVD import PVD_obj, PVD_cons
from Problems.TBT import TBT_obj, TBT_cons
from Problems.WBD import WBD_obj, WBD_cons
import numpy as np
import warnings
warnings.filterwarnings("ignore")


FITNESS_MAX = 1e50
Obj_Func = None
Cons_Func = None

def isNegative(cons):
    for con in cons:
        if con > 0:
            return False
    return True


def Evaluation(indi):
    global Obj_Func, Cons_Func
    obj = Obj_Func(indi)
    cons = Cons_Func(indi)
    if isNegative(cons) == False:
        obj = FITNESS_MAX
    return obj


def main():
    global Obj_Func, Cons_Func
    trials = 30
    PopSize = 100

    Dims = [3, 4, 2, 4]
    problem_names = ["CSD", "PVD", "TBT", "WBD"]
    obj_funcs = [CSD_obj, PVD_obj, TBT_obj, WBD_obj]
    cons_funcs = [CSD_cons, PVD_cons, TBT_cons, WBD_cons]

    CSD_range = [[0.05, 0.25, 2], [2, 1.3, 15]]
    PVD_range = [[0, 0, 10, 10], [99, 99, 200, 200]]
    TBT_range = [[0, 0], [1, 1]]
    WBD_range = [[0.1, 0.1, 0.1, 0.1], [2, 10, 10, 2]]
    scale_range = [CSD_range, PVD_range, TBT_range, WBD_range]

    for i in range(len(Dims)-1):
        SuiteName = "Engineer"
        problem_dict = {
            "fit_func": Evaluation,
            "lb": scale_range[i][0],
            "ub": scale_range[i][1],
            "minmax": "min",
            "log_to": None
        }
        MaxFEs = 20000
        MaxIter = int(MaxFEs / PopSize)

        Obj_Func = obj_funcs[i]
        Cons_Func = cons_funcs[i]

        All_Trial_Best = []
        for j in range(trials):
            hgs_solver = HGS.OriginalHGS(epoch=MaxIter, pop_size=PopSize)
            np.random.seed(2022 + 88 * j)
            best_position, best_fitness_value = hgs_solver.solve(problem_dict)
            All_Trial_Best.append(hgs_solver.history.list_global_best_fit)
        np.savetxt("./HGS_Data/" + SuiteName + "/" + problem_names[i] + ".csv", All_Trial_Best, delimiter=",")


if __name__ == "__main__":
    if os.path.exists('./HGS_Data/Engineer') == False:
        os.makedirs('./HGS_Data/Engineer')
    main()