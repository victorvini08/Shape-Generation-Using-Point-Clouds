import numpy as np

def create_input_points_array(file_location):
    points=[]
    with open(file_location) as myfile:
        lines = myfile.readlines()[10:]
    for line in lines:
        coordinates = line.split(' ')
        coordinates=np.array(coordinates,dtype=np.float)
        points.append(coordinates)
    points = np.vstack(points)
    return points
    