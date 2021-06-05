import os
from sklearn.neighbors import KDTree
from read_input import create_input_points_array

def kd_tree_building(base_file_location):

	P=[]
	for myfile in os.listdir(base_file_location):
	    file_location = base_file_location + myfile
	    points = create_input_points_array(file_location)
	    #print(points)
	    kd_tree = KDTree(points)
	    sorted_points = points[kd_tree.get_arrays()[1]]
	    P.append(sorted_points.flatten())      
	P = np.vstack(P)
	#print(P.shape)
	return P    