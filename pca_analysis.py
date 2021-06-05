from sklearn.decomposition import PCA

def pca_analysis(P):
	pca = PCA(100)
	pca.fit(P)
	U = pca.components_
	V = pca.transform(P)
	print(U)
	print(V)

	post_pca_data = pca.transform(P)

	return post_pca_data
	          