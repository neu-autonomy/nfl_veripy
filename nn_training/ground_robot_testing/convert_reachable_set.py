from gettext import find
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import itertools
from itertools import permutations
import pdb

def convert_reachable_set(linearized_reachable_set, theta_range, u_ranges):
    print('yo')

def find_extreme_omegas(reachable_set_pair, u_limits):
    init_set = reachable_set_pair[0]
    reachable_set = reachable_set_pair[1]
    u_min = u_limits[:,0]
    u_max = u_limits[:,1]
    

    min_x_delta = np.min(
        (reachable_set[0,1]-init_set[0,1],
        reachable_set[0,0]-init_set[0,1],
        reachable_set[0,1]-init_set[0,0],
        u_min[0])
    )
    min_y_delta = np.min(
        (reachable_set[1,1]-init_set[1,1],
        reachable_set[1,0]-init_set[1,1],
        reachable_set[1,1]-init_set[1,0],
        u_min[1])
    )
    max_x_delta = np.max(
        (reachable_set[0,1]-init_set[0,1],
        reachable_set[0,0]-init_set[0,1],
        reachable_set[0,1]-init_set[0,0],
        u_max[0])
    )
    max_y_delta = np.max(
        (reachable_set[1,1]-init_set[1,1],
        reachable_set[1,0]-init_set[1,1],
        reachable_set[1,1]-init_set[1,0],
        u_max[1])
    )
    if max_x_delta > 0 and min_x_delta < 0:
        min_x_delta = 0
    min_x_dist = np.min((np.abs(min_x_delta), np.abs(max_x_delta)))
    max_x_dist = np.max((np.abs(min_x_delta), np.abs(max_x_delta)))
    min_y_dist = np.min((np.abs(min_y_delta), np.abs(max_y_delta)))
    max_y_dist = np.max((np.abs(min_y_delta), np.abs(max_y_delta)))

def rotate_frame(reachable_set_pair, theta):
    transformed_pair = np.zeros(reachable_set_pair.shape)
    R = np.array(
        [
            [np.cos(theta), np.sin(theta)],
            [-np.sin(theta), np.cos(theta)]
        ]
    )
    for i, rect in enumerate(reachable_set_pair):
        corners = []
        permut = permutations(rect[0], len(rect[1]))
        for comb in permut:
            zipped = zip(comb, rect[1])
            corners.append(list(zipped))
        corners = np.array(corners).reshape(4,2).T
        transformed_corners = R @ corners
        xmin = np.min(transformed_corners[0])
        xmax = np.max(transformed_corners[0])
        ymin = np.min(transformed_corners[1])
        ymax = np.max(transformed_corners[1])
        # pdb.set_trace()
        # test = combinations(rect[0],rect[1])
        transformed_pair[i] = np.array(
            [
                [xmin, xmax],
                [ymin, ymax]
            ]
        )
        
    
    return transformed_pair
   
def plot_reachable_set(reachable_set):
    fig, ax = plt.subplots()
    #ax.plot([0,10],[0,10])
    for rect in reachable_set:
        dims = rect[:,1]-rect[:,0]
        ax.add_patch(
            Rectangle(
                rect[:,0], dims[0], dims[1],
                edgecolor = 'blue',
                fill = False
            )
        )
    ax.set_xlim([-1,8])
    ax.set_ylim([-1,8])
    plt.show()



    


def main():
    reachable_set = np.array(
        [
            [
                [0,1],
                [0,1]
            ],
            [
                [0,1],
                [3,4]
            ],
            [
                [4,5],
                [5,6]
            ]
        ]
    )
    plot_reachable_set(reachable_set)
    transformed_set = rotate_frame(reachable_set[0:2], np.pi/4)
    find_extreme_omegas(transformed_set)
    final = np.append(transformed_set, [rotate_frame(reachable_set[1:3], np.pi/4)[1]], axis=0)
    # pdb.set_trace()
    plot_reachable_set(final)
    # convert_reachable_set()

if __name__ == "__main__":
    main()
