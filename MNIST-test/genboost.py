import pygmo as pg
import copy
class genboost:
    def __init__(self, problem):
        self.problem = problem
        
    def run(self, params : dict):
        self.params = copy.copy(params)
        self.prob = pg.problem(self.problem)
        self.algo_name = self.params.pop('algo')
        self.ind_num = self.params.pop('ind_num')
        self.algo = pg.algorithm(pg.__dict__[self.algo_name](**self.params))
        self.algo.set_verbosity(1)
        self.pop = pg.population(self.prob, self.ind_num)
        self.pop = self.algo.evolve(self.pop)
        self.log = self.algo.extract(pg.__dict__[self.algo_name]).get_log()
        return self.pop

