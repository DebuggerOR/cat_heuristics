from abc import ABC, abstractmethod

import numpy as np
from scipy.special import softmax

from utils import NUM_TRIES_NEIGHBORS, EXPLOIT_PARAM


class SearchAlgo(ABC):
    def __init__(self, init_state, max_iters=100):
        self.init_state = init_state
        self.max_iters = max_iters
        super().__init__()

    @abstractmethod
    def run(self):
        pass


class HillClimbing(SearchAlgo):
    def run(self):
        print('*** running hill_climbing ***')
        current = self.init_state

        for i in range(self.max_iters):
            changed = False
            min_val = current.evaluate()

            for j in range(NUM_TRIES_NEIGHBORS):
                neighbors = current.get_neighbors(i+1)

                for n in neighbors:
                    cur_val = n.evaluate()
                    if cur_val < min_val:
                        min_val = cur_val
                        current = n
                        changed = True

            if not changed:
                i -= 1
        return current


class SimulatedAnnealing(SearchAlgo):
    def run(self):
        print('*** running simulated_annealing ***')
        current = self.init_state

        for i in range(self.max_iters):
            temp = 1 / np.log(i + 2)
            if np.random.uniform(0, temp) > EXPLOIT_PARAM:
                neighbors = current.get_neighbors(i + 1)
                n = np.random.choice(neighbors, 1, p=softmax([n.evaluate() for n in neighbors]))[0]
                current = n
            else:
                min_val = current.evaluate()
                for j in range(NUM_TRIES_NEIGHBORS):
                    neighbors = current.get_neighbors(i + 1)

                    for n in neighbors:
                        cur_val = n.evaluate()
                        if cur_val < min_val:
                            min_val = cur_val
                            current = n

        return current