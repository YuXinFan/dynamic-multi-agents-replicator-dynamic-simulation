import numpy as np
import matplotlib.pyplot as plt
import math
import scipy.optimize as opt

print("Predefine value define")
# list of object type, in fixed order, contains [type1,..., typen]
object_type_list = []
# list of object percent in fixed order
object_percent_list = []

# list of list, in fixed order, contains [strategy_list_of_object1, strategy_list_of_object2, ..., strategy_list_of_objectn]
strategy_type_list = []
# list of list, in fixed order, contains [strategy_percent_list_of_obj1, ...,strategy_percent_list_of_objn]
strategy_percent_list = []

# parameter list, a list of list of [c, w1, w2]
parameter_list = []
payoff_table = {}
opt_func_list = []

# threshold of delete or add new strategy
threshold = 0.1
# append lita, zisi, breaker
opt_func_list.append(lambda x,i,j: -(x[1]*(2*parameter_list[i][j][0] - 2 * parameter_list[i][j][2]) - 2 * parameter_list[i][j][2]*math.cos(math.pi * x[1]) + x[2]*(-2*math.cos(parameter_list[i][j][1]*math.pi)-2*parameter_list[i][j][1]) + x[0]*(2+2*parameter_list[i][j][1])+2*parameter_list[i][j][0]))
opt_func_list.append(lambda x,i,j: -(x[1]*(-parameter_list[i][j][1]+parameter_list[i][j][2]) - 3 *  parameter_list[i][j][2]*math.cos(math.pi*x[1]) + x[2]*(-3*parameter_list[i][j][1]+math.cos(math.pi * parameter_list[i][j][1]))+x[0]*(3*parameter_list[i][j][1]-1) + 3 * parameter_list[i][j][0]))
opt_func_list.append(lambda x,i,j: -(x[1]*(-4*parameter_list[i][j][0] + 4*parameter_list[i][j][2]) -4*parameter_list[i][j][2]*math.cos(x[1]*math.pi)+x[2]*(4*math.cos(parameter_list[i][j][1]*math.pi)-4*parameter_list[i][j][1])+x[0]*(-4+4*parameter_list[i][j][1])+4*parameter_list[i][j][0]))
print("Help function")
# normalize the interval to [0,1]
def normalize(x, xmax, xmin,ymax = 1, ymin = 0):
    return ymin + (ymax-ymin)/(xmax-xmin) * (x-xmin)

# Cut the tail of float point vector value x
def cutTail(x):
    t = x
    for i in range(0, len(x)):
        if (t[i] < 0):
            t[i] = 0
        elif (t[i] > 1):
            t[i] = 1
        t[i] = round(t[i] , 4)
    return t

# add new strategy to an objective
def appendNewStrategy(object_type_, parameter_):
    global strategy_percent_list, strategy_type_list, parameter_list
    if ( list(parameter_) in list((parameter_list[object_type_]))):
        return 
    else:
        strategy_type_list[object_type_].append(len(strategy_type_list[object_type_]))

        strategy_percent_list_of_this_obj = strategy_percent_list[object_type_]
        max_percent_idx =  strategy_percent_list_of_this_obj.index(max(strategy_percent_list_of_this_obj))
        strategy_percent_list[object_type_][max_percent_idx] -= threshold
        strategy_percent_list[object_type_].append(threshold)
        parameter_list[object_type_].append(list(parameter_))

# generate a all 0 list with given length
def generateZeroList(len):
    return [0 for i in range(0, len)]

print("Function define")
# map of (action this, action others) -> (payoff this, payoff others)
def initPayoffTable(a = 2, b = 3, c = 1, d = 4):
    payoff_table[(0,0)] =(a, a) 
    payoff_table[(0,1)] =(b, c) 
    payoff_table[(1,1)] =(d, d)
    payoff_table[(1,0)] =(c, b)

def initObjectPopulation(object_number, *object_percent):
    global object_type_list, object_percent_list

    object_type_list = [i for i in range(0, object_number)]
    object_percent_list_tmp = list(object_percent)
    object_percent_list = [i for i in object_percent_list_tmp]

#strategy distribution from index order
def initStrategyPopulation(strategy_number, strategy_percent):
    x_vector = strategy_percent
    x = x_vector#np.asarray(x_vector)
    if ( (len(x_vector) == strategy_number) and (sum(x_vector) == 1)):        
        return x
    else:
        print("Unmatching strategy_number and number of strategies.")
        return
    return
# parameter [c, w1, w2] of all strategy of an object
def initParemeter(strategy_number, parameter_list_set):
    return parameter_list_set

def f1(cy, w2y):
    return cy-w2y
def f2(w1y):
    return -math.cos(w1y * math.pi)

# TO BE DEFINE
def learnNewStrategy(object_type_):
    return

def getParemeter(object_type_, strategy_type_):
    global parameter_list
    return parameter_list[object_type_][strategy_type_]

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
# payoff of action pair
def payoff(object_type_1, strategy_type_1, object_type_2, strategy_type_2):
    ret_pair = action(object_type_1, strategy_type_1, object_type_2, strategy_type_2)
    return (-(ret_pair[0]-ret_pair[1]) + (ret_pair[0] + ret_pair[1]))
# outcome of diff object
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


# fitness of one strategy of an objective
def fitness(object_type_,strategy_type_, strategy_percent_list_ ):
    ret = 0
    for i in range(0, len(strategy_type_list)):
        for j in range(0, len(strategy_type_list[i])):
            percent_of_another = strategy_percent_list_[i][j]*object_percent_list[i]
            play_out = outcome(object_type_, strategy_type_, i, j)
            ret += play_out * percent_of_another
    return ret

def averageFitness(object_type_, strategy_percent_list_):
    ret = 0
    strategy_percent_list_of_this_object = strategy_percent_list_[object_type_]
    for i in range(0, len(strategy_percent_list_of_this_object)):
        ret += strategy_percent_list_of_this_object[i] * fitness(object_type_, i, strategy_percent_list_)
    return ret

def replicate(object_type_, strategy_type_, strategy_percent_list_):
    ret = 0
    strategy_percent_of_this_object = strategy_percent_list_[object_type_][strategy_type_]
    ret =  strategy_percent_of_this_object * (fitness(object_type_, strategy_type_, strategy_percent_list_) - averageFitness(object_type_, strategy_percent_list_))
    return ret




print("Main Simulate Program")
initPayoffTable()
# Define type 0->lita zhuyi, type 1->self-interested, type 2->breaker
initObjectPopulation(3, 0.1, 0.1, 0.8)

#define strategy population
strategy_number_set = [2,2,2]
strategy_percent_set = [[0.5, 0.5],
                        [0.5, 0.5],
                        [0.5, 0.5]]
                        # C , W1, W2
                        
parameter_set =  [[[0.3, 0.1, 0.1], [0.5, 0.1, 0.1]], [[0.5, 0.1, 0.5], [0.1, 0.1, 0.9]], [[0.0, 0.1, 1.0], [0.1, 0.1, 0.9]]]

for i in range(0, len(object_type_list)):
    strategy_percent_list.append(initStrategyPopulation(strategy_number_set[i], strategy_percent_set[i]))
    strategy_type_list.append([j for j in range(0, len(strategy_percent_list[i]))])

parameter_list = parameter_set
print("Init object percent list: \n", object_percent_list)
print("Init strategy percent list: \n", strategy_percent_list)
print("Init parameter list: \n", parameter_list)
print("-------------------------------")
delta = 0.5
# list of list of list record fitness of each strategy in each iteration
fitness_list = [[[] for j in range(0, len(strategy_percent_list[i]))] for i in range(0, len(object_type_list))]

# list of list of list record percent of each strategy in each iteration
percent_list = [[[] for j in range(0, len(strategy_percent_list[i]))] for i in range(0, len(object_type_list))]
# define optmize constraint
cons = ({'type':'ineq', 'fun':lambda x: x[0]},
        {'type':'ineq', 'fun':lambda x: 1-x[0]},
        {'type':'ineq', 'fun':lambda x: x[1]},
        {'type':'ineq', 'fun':lambda x: 1-x[1]},
        {'type':'ineq', 'fun':lambda x: x[2]},
        {'type':'ineq', 'fun':lambda x: 1-x[2]},
       )
iter_number=2000
for i in range(0, iter_number):
    # Append information to list for draw curve
    # look through Objec type 
    for j in range(0, len(object_type_list)):
        # look through stretagy type
        for k in range(0, len(strategy_type_list[j])):
            #fitness_list[j][k].append(fitness(j, k, strategy_percent_list)*delta)
            percent_list[j][k].append(strategy_percent_list[j][k])
    # generate new strategy for each objective
    for k in range(0, len(object_type_list)):
        opt_func_object = opt_func_list[k]
        def opt_func(x):
            l = np.array([ opt_func_object(x,i,j) for i in range(0, len(parameter_list)) for j in range(0, len(parameter_list[i]))])
            return sum(l)
        init_vector = [1,1,1]
        result = opt.minimize(opt_func, init_vector,constraints=cons)
        cuted_x = cutTail(result.x)
        #print("result", cuted_x)
        # TO DO add new strategy to space
        appendNewStrategy(k, cuted_x)
        #print("new strategy percent ", strategy_percent_list)
        #print("new paremeter list ", parameter_list)
        # append zero list to recorded list
        #fitness_list[k].append(generateZeroList(i+1))
        percent_list[k].append(generateZeroList(i+1))

    # Do replicator
    tmp_strategy_percent_list = strategy_percent_list
    for j in range(0, len(object_type_list) ):
        for k in range(0, len(strategy_type_list[j])):
            strategy_percent_list[j][k] =  strategy_percent_list[j][k] + replicate(j, k, tmp_strategy_percent_list) * delta
print("Final object percent list: \n", object_percent_list)
print("Final strategy percent list: \n", strategy_percent_list)
print("Final parameter list: \n", parameter_list)
#print("strategy_percent_list", percent_list)
#print("fitness_list", fitness_list)
for i in range(0, len(object_type_list)):
    plt.subplot(1,len(object_type_list),i+1)
    for j in range(0, len(strategy_type_list[i])):
        lable_percent = "strategy " + str(parameter_list[i][j])
        plt.plot(percent_list[i][j],  label =lable_percent)
        plt.grid()
        plt.ylim(-0.1, 1.1)
        plt.xlim(0, iter_number+20)
        plt.legend(loc='best')
    # plt.subplot(2,len(object_type_list),len(object_type_list)+i+1)
    # for j in range(0, len(strategy_type_list[i])):
    #     lable_fitness = "fitness of strategy" + str(j)
    #     plt.plot(fitness_list[i][j],  label =lable_fitness)
    #     plt.grid()
    #     plt.ylim(-0.1, 2.1)
    #     plt.xlim(0, iter_number+20)
    #     plt.legend(loc='best')
plt.show()


    # fitness_list_a.append(self.fitness(0, self.x)*delta)
    # fitness_list_b.append(self.fitness(1, self.x)*delta)
    # x_a.append(x[0])
    # x_b.append(x[1])
    # average_fitness_list.append(self.averageFitness(self.x)*delta)

    
    # tmp_x = x
    # x[0] = x[0] + self.replicate(0, tmp_x)*delta
    # x[1] = x[1] + self.replicate(1, tmp_x)*delta

