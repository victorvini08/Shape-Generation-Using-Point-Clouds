import argparse
from kd_tree_building import kd_tree_building
from gan_training import train_gan_model
from pca_analysis import pca_analysis
from iterative_point_ordering import iterative_PCA
def main():

	parser = argparse.ArgumentParser()

	parser.add_argument("-b","--base_file",help="Base File Location of dataset")
	parser.add_argument("-e","--epochs",help="Number of epochs")
	args=parser.parse_args()
	base_file_location = args.base_file

	P = kd_tree_building(base_file_location)

	#post_pca_data = pca_analysis(P)
	post_pca_data = iterative_PCA(1000,10000,P)
	
	train_gan_model(args.epochs,post_pca_data)

if __name__ == '__main__':
    main()