from abc import abstractmethod, ABC

import numpy as np
from numpy import average
from scipy.stats import norm
from skimage.draw import circle
from skimage.io import imshow

COLORS = []
NUM_TRIES_NEIGHBORS = 10
NUM_TRIES_ALGO = 5
EXPLOIT_PARAM = 0.5


def create_colors(img_mat, colors_num=10, epsilon=300):
    print('*** creating colors ***')
    colors = []
    num_rows, num_cols, _ = img_mat.shape

    while len(colors) < colors_num:
        rr = np.random.randint(num_rows)
        rc = np.random.randint(num_cols)
        sample = img_mat[rr][rc]

        is_new = True
        for j in range(len(colors)):
            if np.sum((colors[j][:] - sample[:]) ** 2) < epsilon:
                is_new = False
        if is_new:
            colors.append(sample)

    global COLORS
    COLORS = colors
    imshow(np.array([colors]))


class State(ABC):
    def __init__(self, value, goal_value):
        self.value = value
        self.goal_value = goal_value
        super().__init__()

    def get_value(self):
        return self.value

    @abstractmethod
    def get_neighbors(self, iter):
        pass

    @abstractmethod
    def evaluate(self):
        pass


class ImageState(State):
    def get_neighbors(self, iter):
        neighbors = []
        rows, cols, _ = self.value.shape
        radious = int(min(rows, cols) / 2) / (np.log(iter + 1))

        rr = np.random.randint(rows)
        rc = np.random.randint(cols)
        for col in COLORS:
            c1, c2 = circle(rr, rc, radious, shape=(rows, cols))
            val = np.copy(self.value)
            val[c1, c2] = col
            neighbors.append(ImageState(val, self.goal_value))

        return neighbors

    def evaluate(self):
        err = np.sum((self.value.astype("float") - self.goal_value.astype("float")) ** 2)
        err /= (float(self.value.shape[0] * self.value.shape[1] * self.value.shape[2]) * 10000)
        return err
