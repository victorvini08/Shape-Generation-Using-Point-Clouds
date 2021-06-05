from sklearn.decomposition import PCA
import numpy as np

def iterative_PCA(iterations,K,P):
	pca = PCA(100)
	for i in range(iterations):
	    pca.fit(P)
	    for j in range(K):
	        p1,p2=np.random.randint(0,1000,2)
	        shape=np.random.randint(0,5000,1)
	        initial_loss=((pca.inverse_transform(pca.transform(P[shape]))-P[shape])**2).sum()
	        swapped_loss = ((pca.inverse_transform(pca.transform(P[shape]))-P[shape])**2).sum()

	        if swapped_loss > initial_loss:
	            P[shape,3*p1 : 3*p1 +3] = P[shape,3*p2 : 3*p2 +3]
	            P[shape,3*p2 : 3*p2 +3]= P[shape,3*p1 : 3*p1 +3]
	post_pca_data = pca.transform(P)
    return post_pca_data