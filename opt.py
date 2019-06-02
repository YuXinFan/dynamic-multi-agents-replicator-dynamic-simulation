import numpy as np
from scipy.optimize import minimize
import math

parameter_list = [[[1, 0.9, 0.1], [0.9, 0.1, 0.1]],
                 [[0.5, 0.5, 0.5], [0.9, 0.9, 0.9]],
                 [[0.0, 1.0, 1.0], [0.1, 0.9, 0.9]]]
def normalize(x, xmax, xmin,ymax = 1, ymin = 0):
    return ymin + (ymax-ymin)/(xmax-xmin) * (x-xmin)
def f1(cy, w2y):
    return cy-w2y
def f2(w1y):
    return -math.cos(w1y * math.pi)
def getParemeter(object_type_, strategy_type_):
    global parameter_list
    return parameter_list[object_type_][strategy_type_]

def optfunc(x):
    f = lambda x,i,j: -(x[1]*(-parameter_list[i][j][1]+parameter_list[i][j][2]) - 3 *  parameter_list[i][j][2]*math.cos(math.pi*x[1]) + x[2]*(-3*parameter_list[i][j][1]+math.cos(math.pi * parameter_list[i][j][1]))+x[0]*(3*parameter_list[i][j][1]-1) + 3 * parameter_list[i][j][0])
    a = np.array([ f(x,i,j) for i in range(0, len(parameter_list)) for j in range(0, len(parameter_list[i]))])
    return sum(a)
#  action of two player
def action(object_type_1, strategy_type_1, object_type_2, strategy_type_2):
    parameter_1 = getParemeter(object_type_1, strategy_type_1)
    parameter_2 = getParemeter(object_type_2, strategy_type_2)
    ret_1 = parameter_1[1]*f1(parameter_2[0], parameter_2[2]) + parameter_2[2]*f2(parameter_2[1]) + parameter_1[0]
    ret_2 = parameter_2[1]*f1(parameter_1[0], parameter_1[2]) + parameter_1[2]*f2(parameter_1[1]) + parameter_2[0]
    ret_1 = normalize(ret_1, 3, -2)
    ret_2 = normalize(ret_2, 3, -2)
    # if (ret_1 > 1):
    #     ret_1 = 1
    # elif (ret_1 < 0):
    #     ret_1 = 0
    
    # if (ret_2 > 1):
    #     ret_2 = 1
    # elif (ret_2 < 0):
    #     ret_2 = 0 
    return (ret_1, ret_2)
def outcome(object_type_1, strategy_type_1, object_type_2, strategy_type_2):
    ret_pair = action(object_type_1, strategy_type_1, object_type_2, strategy_type_2)
    # lita zhuyi
    ret = 0
    if (object_type_1 == 0):
        ret = 2 * ret_pair[0] + 2 * ret_pair[1]
        ret = normalize(ret, 4, 0)
        #print(ret)
    elif (object_type_1 == 1):
        ret = - ret_pair[0] + 3 * ret_pair[1]
        ret = normalize(ret, 3, -1)
    else:
        ret = - 4 * ret_pair[0] + 4 * ret_pair[1]
        ret = normalize(ret, 4, -4)
    return ret

# def fun_rosenbrock(x):

#     return np.array([2 * ret_pair[0] + 2 * ret_pair[1] for i in range(0,3) for j in range(0, 2) ])

from scipy.optimize import least_squares
input = np.array([1, 1, 1])
cons = ({'type':'ineq', 'fun':lambda x: x[0]},
        {'type':'ineq', 'fun':lambda x: 1-x[0]},
        {'type':'ineq', 'fun':lambda x: x[1]},
        {'type':'ineq', 'fun':lambda x: 1-x[1]},
        {'type':'ineq', 'fun':lambda x: x[2]},
        {'type':'ineq', 'fun':lambda x: 1-x[2]},
       )
#res = least_squares(optfunc, input,constraints=cons)
res = minimize(optfunc, input,constraints=cons)

print (res.x)

# print(optfunc([1,1,1]))