from skimage.io import imread, imsave, imshow
from skimage.draw import circle
from abc import ABC, abstractmethod
import numpy as np

COLORS = []

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
        rows,cols,_ = self.value.shape
        radious = int(min(rows,cols)/2) / (np.log(iter + 1))

        num_rows, num_cols, _ = img_mat.shape
        rr = np.random.randint(num_rows)
        rc = np.random.randint(num_cols)
        for col in COLORS:
            c1, c2 = circle(rr, rc, radious, shape=(num_rows,num_cols))
            val = np.copy(self.value)
            val[c1, c2] = col
            neighbors.append(ImageState(val,self.goal_value))

        return neighbors

    def evaluate(self):
        return np.sum((self.value[:, :, 0:3] - self.goal_value[:, :, 0:3]) ** 2)


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
            neighbors = current.get_neighbors(i)

            min_val = current.evaluate()
            for n in neighbors:
                cur_val = n.evaluate()
                if cur_val < min_val:
                    min_val = cur_val
                    current = n

            # imshow(min_neighbor)
            # plt.show()

        return current


class SimulatedAnnealing(SearchAlgo):
    def run(self):
        current = self.init_state
        worst_eval = current.evaluate()

        for i in range(self.max_iters):
            neighbors = current.get_neighbors(i)

            cur_val = current.evaluate()
            n = np.random.choice(neighbors)
            n_val = n.evaluate()
            if cur_val > n_val:
                current = n
            else:
                p = 1 - (n_val / worst_eval)
                p = p / (np.log(i + 1))
                if np.random.random() < p:
                    current = n
                else:
                    i-=1

        return current


def create_colors(img_mat, colors_num=8, epsilon=300):
    colors = []
    num_rows,num_cols,_ = img_mat.shape

    while len(colors) < colors_num:
        rr = np.random.randint(num_rows)
        rc = np.random.randint(num_cols)
        sample = img_mat[rr][rc]
        is_new = True
        for j in range(len(colors)):
            if np.sum((colors[j][:]-sample[:])**2) < epsilon:
                is_new = False
        if is_new:
            colors.append(sample)

    imshow(np.array([colors]))
    return colors


if __name__ == '__main__':
    img_mat = imread('cat.jpg')
    COLORS = create_colors(img_mat)
    goal_state = ImageState(img_mat, img_mat)

    init_mat = np.full(img_mat.shape, 255)
    init_state = ImageState(init_mat, img_mat)

    algo = HillClimbing(init_state, 10000)
    fin_state = algo.run()

    imsave('hill_cat.png', fin_state.get_value())

    algo = SimulatedAnnealing(init_state, 10000)
    fin_state = algo.run()

    imsave('sim_cat.png', fin_state.get_value())

