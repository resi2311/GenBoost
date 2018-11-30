import pygmo as pg
import copy
class genboost:
    def __init__(self,problem):
        self.problem = problem
        
    def run(self,params : dict):
        self.params = copy.copy(params)
        self.prob = pg.problem(self.problem)
        num_ind = self.params.pop('num_ind')
        self.algo = pg.algorithm(pg.sga(**self.params))
        self.pop = pg.population(self.prob,num_ind)
        self.pop = self.algo.evolve(self.pop)
        return self.pop

