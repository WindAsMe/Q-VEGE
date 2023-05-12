"""
Hadi Bayzidi, Siamak Talatahari, Meysam Saraee, et al.
Social Network Search for Solving Engineering Optimization Problems[J].
Computational Intelligence and Neuroscience
"""
import numpy as np

l = 100
P = 2
sigma = 2

"""
Three-Bar Truss
"""
def TBT_obj(X):
    """
    :param X:
    0 <= X[0],
         X[1] <= 1
    :return:
    """
    return (2 * np.sqrt(2) * X[0] + X[1]) * l


def TBT_cons(X):
    con1 = (np.sqrt(2) * X[0] + X[1]) / (np.sqrt(2) * X[0] ** 2 + 2 * X[0] * X[1]) * P - sigma
    con2 = X[1] / (np.sqrt(2) * X[0] ** 2 + 2 * X[0] * X[1]) * P - sigma
    con3 = 1 / (np.sqrt(2) * X[1] + X[0]) * P - sigma
    return [con1, con2, con3]

