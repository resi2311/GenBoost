import numpy as np
class Problem:
    def __init__(self, fit_func, dim=407050, lb=-1., rb=1.):
        self.dim = dim
        self.lb = lb
        self.rb = rb
        self.fit_func = fit_func
        
    def fitness(self, weights):
        return self.fit_func(weights)

    def get_bounds(self):
        return (np.full((self.dim,), self.lb), np.full((self.dim,), self.rb))