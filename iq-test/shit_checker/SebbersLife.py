import numpy as np
import pprint

Blank = 0
Red = 10
Blue = 100
Green = 1000
Yellow = 10000

world = np.array([[0, 0, 0, 0], [0, 0, 10, 0], [0, 100, 0, 10], [0, 0, 1000, 0]])

pprint.pprint(world)
size = world.shape[0]

neighbor = np.zeros(world.shape, dtype=int)
neighbor[1:] += world[:-1]  # North
neighbor[:-1] += world[1:]  # South
neighbor[:, 1:] += world[:, :-1]  # West
neighbor[:, :-1] += world[:, 1:]  # East

pprint.pprint(neighbor)


# Combinations
Red_combi = [10, 20, 30, 40, 120, 130, 1020, 1030, 1120, 10020, 10030]
Blue_combi = [100, 200, 300, 400, 210, 310, 1200, 1300, 10200, 10300]
Green_combi = [1000, 2000, 3000, 4000, 2010, 3010, 2100, 3100, 12000, 13000]
Yellow_combi = [10000, 20000, 30000, 40000, 20010, 30010, 20100, 30100, 21000, 31000]
Blank_combi = [
    0,
    110,
    220,
    1010,
    2020,
    1100,
    2200,
    10010,
    20020,
    10100,
    20200,
    11000,
    22000,
]

neighbor[np.isin(neighbor, Red_combi)] = Red
neighbor[np.isin(neighbor, Blue_combi)] = Blue
neighbor[np.isin(neighbor, Green_combi)] = Green
neighbor[np.isin(neighbor, Yellow_combi)] = Yellow
neighbor[np.isin(neighbor, Blank_combi)] = Blank

pprint.pprint(neighbor)


def next_state(world):
    size = world.shape[0]
    neighbors = np.zeros(shape=(size, size), dtype=int)
    new_world = np.zeros(shape=(size, size), dtype=int)
    neighbor_count = 0
    # Ignore edges: start xrange: in 1
    for rows in range(1, size - 1):
        for cols in range(1, size - 1):
            # Check neighbors
            for i in [-1, 0, 1]:
                for j in [-1, 0, 1]:
                    # Condition to not count existing cell.
                    if rows + i != rows or cols + j != cols:
                        neighbor_count += world[rows + i][cols + j]
                        neighbors[rows][cols] = neighbor_count

            # THIS STUFF <3 <3 <3
            # if neighbors[rows][cols] == 3 or (world[rows][cols] == 1 and neighbors[rows][cols] == 2):
            #   new_world[rows][cols] = 1
            if neighbors[rows][cols] in Red_combi:
                new_world
            # elif
            else:
                new_world[rows][cols] = 0
            neighbor_count = 0

    pprint.pprint(neighbors)
    return new_world

