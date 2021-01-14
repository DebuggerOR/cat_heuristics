from skimage.io import imread, imsave, imshow
from skimage.draw import circle
from abc import ABC, abstractmethod
import numpy as np
import matplotlib.colors as mcolors
from matplotlib import colors
import matplotlib.pyplot as plt
import copy

COLORS = []

class State(ABC):
    def __init__(self, value, goal_value):
        self.value = value
        self.goal_value = goal_value
        super().__init__()

    def get_value(self):
        return self.value

    @abstractmethod
    def next_state(self):
        pass


class ImageState(State):
    def next_state(self):
        print('*** start next ***')
        neighbors = []
        rows,cols,_ = self.value.shape
        rad_range = int(min(rows,cols)/4)

        for r in range(100,rows-100,100):
            for c in range(100,cols-100,100):
                for col in COLORS:
                    rd = 100
                    rr, cc = circle(r, c, rd)
                    val = np.copy(self.value)
                    val[rr, cc] = col
                    neighbors.append(val)

        # for r in range(0,rows,100):
        #     for c in range(0,cols,100):
        #         for col in COLORS:
        #             rd = 100
        #             rr, cc = circle(r, c, rd)
        #             val = copy.deepcopy(self.value)
        #             val[rr, cc] = col
        #             neighbors.append(val)
                # for rd in range(int(rad_range/11), rad_range, 10):

        min_neighbor = neighbors[0]
        min_val = np.sum((min_neighbor[:,:,0:3]-self.goal_value[:,:,0:3])**2)
        for n in neighbors:
            cur_val = np.sum((n[:,:,0:3]-self.goal_value[:,:,0:3])**2)
            if cur_val < min_val:
                min_neighbor = n
                min_val = cur_val

        imshow(min_neighbor)
        plt.show()

        self.value = min_neighbor


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
        current = self.init_state

        for i in range(self.max_iters):
            current.next_state()

        return current


class SimulatedAnnealing(SearchAlgo):
    pass


def create_colors():
    cols = []
    #tab_cols = mcolors.TABLEAU_COLORS
    tab_cols = mcolors.BASE_COLORS
    for c in tab_cols.values():
        r,g,b = colors.to_rgb(c)
        r,g,b = int(255*r),int(255*g),int(255*b)
        cols.append([r,g,b])
    return cols


if __name__ == '__main__':
    COLORS = create_colors()

    img_mat = imread('cat1.jpg')
    goal_state = ImageState(img_mat, img_mat)

    init_mat = np.full(img_mat.shape, 255)
    init_state = ImageState(init_mat, img_mat)

    algo = HillClimbing(init_state, 10)
    fin_state = algo.run()

    imsave('my_cat1.png', fin_state.get_value())

