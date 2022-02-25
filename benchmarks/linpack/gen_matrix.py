import compress_pickle
import sys 
from numpy import random, matrix

def generate_matrix(N):
    A=random.random_sample((N,N))-0.5
    B=A.sum(axis=1)

    # Convert to matrices
    A=matrix(A)

    B=matrix(B.reshape((N,1)))
    
    matrices = {'A': A, 'B': B}
    compress_pickle.dump( matrices, open( "matrices-"+ str(N) +".p", "wb" ), compression='gzip' )
    # pickle.dump( matrices, open( "matrices-"+ str(N) +".p", "wb" ) )

N = int(sys.argv[1])
generate_matrix(N)