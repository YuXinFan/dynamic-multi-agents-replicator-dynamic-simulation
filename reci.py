import numpy as np
import matplotlib.pyplot as plt
#2x2payoff table,population replicator dynamic

class replicator2t:
    type_list = [0,1]
    a = 2
    b = 3
    c = 1
    d = 4
    payoff_table = {}
    x = 0
    tmp_x = 0

    def __init__(self):
        self.payoff_table[(0,0)] =(self.a,self.a) 
        self.payoff_table[(0,1)] =(self.b,self.c) 
        self.payoff_table[(1,1)] =(self.d,self.d)
        self.payoff_table[(1,0)] =(self.c,self.b)


    def initPopulation(self, type_number, *numbers):
        x_vector = list(numbers)
        self.x = np.asarray(x_vector)
        if ( (len(x_vector) == type_number) and (sum(x_vector) == 1)):        
            return self.x
        else:
            print("Unmatching type_number and number of parameter numbers.")
            return

    def fitness(self, type, x):
        ret = 0
        ret = self.payoff_table[(type,0)][0] * x[0] + self.payoff_table[(type,1)][0] * x[1] 
        return ret

    def averageFitness(self, x):
        ret = 0
        for i in self.type_list:
            ret += x[i] * self.fitness(i, x)
        return ret

    #replicator equation for one type
    def replicate(self, type, x):
        ret = 0
        ret = x[type] * (self.fitness(type, x) - self.averageFitness(x))
        return ret


    def simulate2t(self):
        x = self.initPopulation(2, 0.4,0.6)
        delta = 0.1
        fitness_list_a = []
        fitness_list_b = []
        x_a = []
        x_b = []
        average_fitness_list = []
        for i in range(100):
            fitness_list_a.append(self.fitness(0, self.x)*delta)
            fitness_list_b.append(self.fitness(1, self.x)*delta)
            x_a.append(x[0])
            x_b.append(x[1])
            average_fitness_list.append(self.averageFitness(self.x)*delta)

            tmp_x = x
            x[0] = x[0] + self.replicate(0, tmp_x)*delta
            x[1] = x[1] + self.replicate(1, tmp_x)*delta

        plt.plot(x_a, 'r', label ='share of strategy A')
        plt.plot(x_b, 'b', label = 'share of strategy B')
        plt.plot(fitness_list_a, 'r--', label ='fitness of strategy A')
        plt.plot(fitness_list_b, 'b--', label = 'fitness of strategy B')
        plt.grid()
        plt.ylim(0, 2)
        plt.xlim(0, 120)
        plt.legend(loc='best')
        plt.show()

print("Main:")
#Set payoff
r = replicator2t()
r.simulate2t()
