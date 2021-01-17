import numpy as np
from skimage.io import imread, imsave
from utils import create_colors, ImageState, NUM_TRIES_ALGO
from algos import HillClimbing, SimulatedAnnealing


def run_hill_climbing(img_mat):
    init_mat = np.full(img_mat.shape, 255)
    init_state = ImageState(init_mat, img_mat)

    fin_state = None
    min_val = np.inf
    for i in range(NUM_TRIES_ALGO):
        cur_fin_state = HillClimbing(init_state, 100).run()
        cur_val = cur_fin_state.evaluate()
        if cur_val < min_val:
            fin_state = cur_fin_state
            min_val = cur_val

    imsave('hill_hommer.png', fin_state.get_value())


def run_sim_annealing(img_mat):
    init_mat = np.full(img_mat.shape, 255)
    init_state = ImageState(init_mat, img_mat)

    fin_state = None
    min_val = np.inf
    for i in range(NUM_TRIES_ALGO):
        cur_fin_state = SimulatedAnnealing(init_state, 100).run()
        cur_val = cur_fin_state.evaluate()
        if cur_val < min_val:
            fin_state = cur_fin_state
            min_val = cur_val

    imsave('sim_hommer.png', fin_state.get_value())


if __name__ == '__main__':
    img_mat = imread('pics/hommer.jpg')

    # create_colors(img_mat)

    run_hill_climbing(img_mat)
    run_sim_annealing(img_mat)