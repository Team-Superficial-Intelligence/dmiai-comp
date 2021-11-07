import numpy as np
import pprint

Blank = 0
Red = 10
Blue = 100
Green = 1000
Yellow = 10000

world = np.array([[0, 0, 0, 0],
                  [0, 0, 10, 0],
                  [0, 100, 0, 10],
                  [0, 0, 1000, 0]])

pprint.pprint(world)
size = world.shape[0]

neighbor = np.zeros(world.shape, dtype=int)
neighbor[1:] += world[:-1]  # North
neighbor[:-1] += world[1:]  # South
neighbor[:,1:] += world[:,:-1]  # West
neighbor[:,:-1] += world[:,1:]  # East

pprint.pprint(neighbor)

# Combinations
Red = [10,20,30,40,120,130,1020,1030,10020,10030]
Blue = [100,200,300,400,210,310,1200,1300,10200,10300] 
Green = [1000,2000,3000,4000,2010,3010,2100,3100,12000,13000] 
Yellow = [10000,20000,30000,40000,20010,30010,20100,30100,21000,31000]
Blank = [0,110,220,1010,2020,1100,2200,10010,20020,10100,20200,11000,22000] 

Higher and lower images will mean no random overlap
only look at one step forwardj


def next_state(world):
    size = world.shape[0]
    neighbors = np.zeros(shape=(size, size), dtype=int)
    new_world = np.zeros(shape=(size, size), dtype=int)
    neighbor_count = 0
    # Ignore edges: start xrange: in 1
    for rows in xrange(1, size - 1):
        for cols in xrange(1, size - 1):
            # Check neighbors
            for i in [-1, 0, 1]:
                for j in [-1, 0, 1]:
                    # Condition to not count existing cell.
                    if rows + i != rows or cols + j != cols:
                        neighbor_count += world[rows + i][cols + j]
                        neighbors[rows][cols] = neighbor_count


            # THIS STUFF <3 <3 <3 
            if neighbors[rows][cols] == 3 or (world[rows][cols] == 1 and neighbors[rows][cols] == 2):
                new_world[rows][cols] = 1
            else:
                new_world[rows][cols] = 0
            neighbor_count = 0

    pprint.pprint(neighbors)
    return new_world


print next_state(world)













new_mat = mat.copy()
    positions = itertools.product(range(new_mat.shape[0]), range(new_mat.shape[1]))
    for pos in positions:    
        elem = new_mat[pos[0], pos[1]]
        if elem != 0:
            # neighbour 0
            if new_mat[pos[0]-1, pos[0]] == 0 and pos[0]-1 >= 0:
                new_mat[pos[0]-1, pos[0]] = elem   
            elif new_mat[pos[0]-1, pos[0]] != elem and pos[0]-1 >= 0:
                new_mat[pos[0]-1, pos[0]] = 0
            if new_mat[pos[0]+1, pos[0]] == 0 and pos[0]+1 <= mat.shape[0]:
                new_mat[pos[0]+1, pos[0]] = elem   
            elif new_mat[pos[0]+1, pos[0]] != elem and pos[0]+1 <= mat.shape[0]:
                new_mat[pos[0]+1, pos[0]] = 0
            if new_mat[pos[0], pos[1]-1] == 0 and pos[1]-1 <= mat.shape[1]:
                new_mat[pos[0], pos[1]-1] = elem   
            elif new_mat[pos[0], pos[1]-1] != elem and pos[1]-1 <= mat.shape[1]:
                new_mat[pos[0], pos[1]-1] = 0
            if new_mat[pos[0], pos[1]+1] == 0 and pos[1]+1 <= mat.shape[1]:
                new_mat[pos[0], pos[1]+1] = elem   
            elif new_mat[pos[0], pos[1]+1] != elem and pos[1]+1 <= mat.shape[0]:
                new_mat[pos[0], pos[1]+1] = 0
    return new_mat