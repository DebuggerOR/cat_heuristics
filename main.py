from skimage.io import imread
from abc import ABC, abstractmethod
import numpy as np


class State(ABC):
    def __init__(self, value):
        self.value = value
        super().__init__()

    @abstractmethod
    def calc_value(self, goal_state):
        pass

    def get_value(self):
        return self.value


class ImageState(State):
    def calc_value(self, goal_state):
        return np.absolute(self.value - goal_state)


class SearchAlgo(ABC):
    def __init__(self, goal_state, max_iters=100):
        self.goal_state = goal_state
        self.init_state = None
        self.max_iters = max_iters
        self.create_init()
        super().__init__()

    @abstractmethod
    def create_init(self):
        pass

    @abstractmethod
    def get_neighbors(self, current):
        pass

    @abstractmethod
    def choose_neighbor(self, neighbors):
        pass

    @abstractmethod
    def run(self):
        pass


class HillClimbing(SearchAlgo, ABC):
    def create_init(self):
        self.init_state = None

    def run(self):
        current = self.init_state

        for i in range(self.max_iters):
            neighbors = self.get_neighbors(current)
            neighbor = self.choose_neighbor(neighbors)
            if neighbor <= current.get_value():
                return current.get_value()
            current = neighbor

        return current


class SimulatedAnnealing(SearchAlgo):
    pass


def load_image(img_name):
    img_mat = imread(img_name)
    return img_mat


if __name__ == '__main__':
    img_mat = load_image('cat.jpg')
    goal_state = ImageState(img_mat)

    algo = HillClimbing(goal_state, 200)
    algo.run()
