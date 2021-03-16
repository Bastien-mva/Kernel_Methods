import numpy as np
import pandas as pd
import numpy.linalg 
from tqdm import tqdm
import numba
from numba import njit,vectorize, jit
import time
import scipy
import cvxopt
import cvxopt.solvers

cvxopt.solvers.options['show_progress'] = False


#@njit
def to_Kernel_train(X, Kernel, sig2 = 1): 
    length = X.shape[0]
    mat_K = np.zeros((length,length))
    for i in range(length):
        x_i = np.squeeze(X[i])
        for j in range(i,length): 
            x_j = np.squeeze(X[j])
            value = Kernel(x_i,x_j)
            mat_K[i,j] = value
            mat_K[j,i] = value 
    return mat_K

#@njit 
def to_Kernel_test(Xtrain,Xtest,Kernel,sig2=1):
    length_train = Xtrain.shape[0]
    length_test = Xtest.shape[0]
    bimat_K = np.zeros((length_train,length_test))
    for i in range(length_train):
        x_i = np.squeeze(Xtrain[i])
        for j in range(length_test): 
            x_j = np.squeeze(Xtest[j])
            value = Kernel(x_i,x_j)
            bimat_K[i,j] = value
    return bimat_K


#@vectorize
def loss(u): 
    return np.log(1+np.exp(-u))
def sigmoid(u): 
    return 1/(1+np.exp(-u))


def grad_loss(u): 
    return -sigmoid(-u)

def hess_loss(u): 
    return sigmoid(u)*sigmoid(-u)

def J(alpha, y, mat_K, lam):
    n = alpha.shape[0]
    regularizer = lam/2*alpha@mat_K@alpha
    vect = mat_K@alpha
    somme = 1/n*np.sum(loss(y*vect))
    return somme+regularizer
   
def grad_J(alpha, y , mat_K, lam ): 
    n = y.shape[0]
    vect_P_alpha = grad_loss(y*(mat_K@alpha))
    return 1/n*mat_K@(vect_P_alpha*y)+ lam*mat_K@alpha

def hess_J(alpha,y_train, mat_K , lam ):
    n = mat_K.shape[0]
    vect_W = hess_loss(y*(mat_K@alpha))
    return 1/n*mat_K +lam*mat_K

def Kernel_logistic_reg_fit(X, y , mat_K, lam, Niter =20):
    alpha = 0.00000*np.random.randn(X.shape[0])
    #alpha = np.ones(2000)
    mat_K = standardize(mat_K)
    lr = 5
    for i in tqdm(range(Niter)): 
        #inv = np.linalg.inv(hess_J(alpha, mat_K = mat_K))
        #alpha-= lr*inv@grad_J(alpha ,mat_K = mat_K)#, mat_K= mat_K)
        alpha-= lr*grad_J(alpha ,mat_K = mat_K)
        '''
        if i%1 ==0 : 
            print('alpha :', alpha)
            print('J :',J(alpha,mat_K = mat_K))
            print('grad :', grad_J(alpha,mat_K = mat_K))
    print('alpha_end :', alpha)
    print('J_end :',J(alpha,mat_K = mat_K))
    print('grad_end :', grad_J(alpha,mat_K = mat_K))'''
    return alpha




#1e-8 marche bien pour lambda
def fit_KRR(mat_K,lam,y):
    #mat_K = standardize(mat_K) #marche pas si on standardise 
    n = mat_K.shape[0]
    full_mat = mat_K +n*lam*np.eye(n)
    alpha = np.linalg.solve(full_mat,y)
    return alpha


def fit_WKRR(mat_K,vect_W,lam,y): 
    '''
    Compute the Weighted Kernel Redge Regression. the Formula is given in the course. 
    The code is optimized, we do not take the diagonal matrix of the square root of W. Instead, 
    we only compute some np.multiply stuff. 
    
    args : 
    
            mat_K : Kernel Matrix that contains the information in the data (K_ij=K(x_i,x_j))
            vect_W : the vector that contains the weight associated to each sample. here we need that all the 
            coefficient of this vector is 0. Otherwise we won't be able to compute the inverse of the square root
            lam : regularization factor 
            y : the vector we train on 
    
    returns :
            
            the vector alpha that satisfy the formula in the course. 
    alpha then needs to be transformed to a function in order to fit the data.
    '''
    min_W = np.min(vect_W)
    if (min_W < 0) or (min_W == 0) : 
        print('Non invertible Matrix W ')
    n = mat_K.shape[0]
    vect_sqrt_W = np.sqrt(vect_W) # the square root of the original vector
    vect_neg_sqrt_W = 1/vect_sqrt_W # the negative square root of the original vector
    b = np.multiply(vect_sqrt_W,y) 
    big_mat = np.multiply(np.multiply(vect_sqrt_W.reshape(-1,1),mat_K), vect_sqrt_W) +n*lam*np.eye(n)
    A = np.multiply(vect_neg_sqrt_W,big_mat)
    return scipy.linalg.solve(A,b)

def cross_val_split(Xtrain,ytrain, cv):
    idx = np.arange(Xtrain.shape[0])
    np.random.shuffle(idx) # we shuffle the indices to get random samples
    sample_size = Xtrain.shape[0]//cv
    Xtrainsplit = []# a list that wil contain each X_train vector. Each element will be smaller than 
                    #X_train. If cv = 3 for example, the size (on the x axis) will be 2/3 the original size 
    ytrainsplit = []
    Xvalsplit = []
    yvalsplit = []
    for i in range(cv-1): 
        #we add the new indices. Here, takes the original vector and returns the vector without the 
        # indices passes in argument 
        Xtrainsplit.append(np.delete(Xtrain,idx[i*sample_size:(i+1)*sample_size],axis = 0))
        ytrainsplit.append(np.delete(ytrain,idx[i*sample_size:(i+1)*sample_size],axis = 0))
        
        # we add the rest 
        # note that here, we keep the same labels for X ( we do not shuffle independantly X and y)
        Xvalsplit.append( Xtrain[idx[i*sample_size:(i+1)*sample_size],:])
        yvalsplit.append(ytrain[idx[i*sample_size:(i+1)*sample_size]])
    # we add the last round. It is different since we can't take float proportion of an array, 
    # we have to take an integer. So, here we just add what remains. 
    Xtrainsplit.append(np.delete(Xtrain,idx[(cv-1)*sample_size:],axis = 0))
    ytrainsplit.append(np.delete(ytrain,idx[(cv-1)*sample_size:],axis = 0))
    Xvalsplit.append( Xtrain[idx[(cv-1)*sample_size:],:])
    yvalsplit.append(ytrain[idx[(cv-1)*sample_size:]])
    return Xtrainsplit,Xvalsplit,ytrainsplit,yvalsplit


def Kernel_cross_val_split(K_train,ytrain, cv):
    idx = np.arange(K_train.shape[0])
    np.random.shuffle(idx) # we shuffle the indices to get random samples
    sample_size = K_train.shape[0]//cv
    Kerneltrainsplit = []# a list that wil contain each Kernel_train vector. Each element will be smaller than 
                    #K_train. If cv = 3 for example, the size (on the x axis) will be 2/3 the original size 
    ytrainsplit = []
    Kernelvalsplit = []
    yvalsplit = []
    for i in range(cv-1): 
        #we add the new indices. Here, takes the original vector and returns the vector without the 
        # indices passes in argument 
        Kerneltrainsplit.append(np.delete(np.delete(K_train,idx[i*sample_size:(i+1)*sample_size],axis =                                                          0),idx[i*sample_size:(i+1)*sample_size],axis = 1))
        ytrainsplit.append(np.delete(ytrain,idx[i*sample_size:(i+1)*sample_size],axis = 0))
        
        # we add the rest 
        # note that here, we keep the same labels for X ( we do not shuffle independantly X and y)
        Kernelvalsplit.append(np.delete(K_train,idx[i*sample_size:(i+1)*sample_size],axis = 0)[:,idx[i*sample_size:                                         (i+1)*sample_size]])
        yvalsplit.append(ytrain[idx[i*sample_size:(i+1)*sample_size]])
    # we add the last round. It is different since we can't take float proportion of an array, 
    # we have to take an integer. So, here we just add what remains. 
    Kerneltrainsplit.append(np.delete(np.delete(K_train,idx[(cv-1)*sample_size:],axis = 0), idx[(cv-                                                              1)*sample_size:], axis = 1))
    ytrainsplit.append(np.delete(ytrain,idx[(cv-1)*sample_size:],axis = 0))
    Kernelvalsplit.append(np.delete(K_train,idx[(cv-1)*sample_size:],axis = 0)[:,idx[(cv-1)*sample_size:]])
    yvalsplit.append(ytrain[idx[(cv-1)*sample_size:]])
    return Kerneltrainsplit,Kernelvalsplit,ytrainsplit,yvalsplit

class estimator(): 
    def __init__(self , Kernel): 
        self.Kernel = Kernel
        self.Kernel_train = None 
        self.alpha = None
        self.b = 0
        
    def predict_proba(self,Kernel_test): 
        if (self.alpha == None).any()==True  : 
            print("Il faut d'abord fitter les données")
        else : 
            return  sigmoid(self.alpha@Kernel_test + self.b)
    
    def predict(self,K_test): 
        if (self.alpha == None).any()==True : 
            print("Il faut d'abord fitter les données")
        else : 
            prob = self.predict_proba(K_test)
            return prob>0.5
        
    def cross_val(self, K_train,ytrain,cv): 
        mistake = 0
        Kerneltrainsplit,Kernelvalsplit,ytrainsplit,yvalsplit = Kernel_cross_val_split(K_train,ytrain,cv)
        for Ktrain,Kval,ytrain,yval in tqdm(zip(Kerneltrainsplit,Kernelvalsplit, ytrainsplit, yvalsplit)):
            self.fit(Ktrain,ytrain)
            pred = self.predict(Kval)
            mistake+=np.sum(np.abs(pred-yval))
        print('Score : ',1- mistake/K_train.shape[0])
        return 1- mistake/K_train.shape[0]
    
    
    

#lam = 1e-8 est bien
class KRR(estimator): 
    def __init__(self , Kernel, lam = 1e-8): 
        super(KRR, self).__init__(Kernel)
        self.lam = lam 
        
    def fit(self, K_train, y): 
        y_copy = (2*y-1).copy()
        self.Kernel_train = K_train
        self.alpha = fit_KRR(self.Kernel_train, self.lam, y_copy)
    
    def set_parameter(self, lam): 
        self.lam = lam 

        
class SVM(estimator):
    def __init__(self, Kernel, C = 1):
        self.kernel = Kernel
        self.C = C
        self.alpha = None
        self.support_vectors = None
        self.support_Y = None
        
    def fit(self, K, Y):
        n = len(Y)
        #calculate the kernel
        #K = np.apply_along_axis(lambda x1: np.apply_along_axis( lambda x2 : self.kernel(x2, x1), 1, X), 1, X)
    
        lbd = 1
        #C = 1 / (2 * n * lbd)  #ça dépend si on veut gérer C ou lambda

        #take Y as -1 and 1
        label = 2 * Y - 1
        
        
        P = cvxopt.matrix(np.outer(label, label) * K ) 
        q = cvxopt.matrix(-np.ones(n))
        A = cvxopt.matrix(label, (1,n), 'd')
        b = cvxopt.matrix(0.0)
                
        '''Je réécris l'inégalité : 0<=y_i*alpha_i<=C avec C = 1/(2*lambda*n)
        comme: G*alpha<=h avec G=stack(diag(Y),-diag(Y)) et h= [C, ..., C, 0, ..., 0] (n fois C et n fois 0)
        ça revient au même et je crois que le solver devrait fonctionner avec ça, mais j'y arrive pas encore
        '''
        # b <= C
        G1 = np.eye(n)
        h1 = np.ones(n) * self.C
        
        # -b <= 0
        G2 = -np.eye(n)
        h2 = np.zeros(n)
        
        G = cvxopt.matrix(np.vstack((G2, G1)))
        h = cvxopt.matrix(np.hstack((h2, h1)))

        #min_b 1/2 * b.T * diag(Y) * K * diag(Y) * b - b.T * 1 s.t. 0<= b <= C
        solver = cvxopt.solvers.qp(P, q, G, h, A, b)
        
        self.beta = np.ravel(solver['x']) # le alpha ou on garde toutes les coordonnées 
                                                # en comparaison avec le self.alpha ou en garde que quelques-uns 
        
        #Je retire les vecteurs avec un alpha trop petit
        eps = 1e-5
        supportIndices = np.abs(self.beta) < eps
        ind = np.arange(n)[supportIndices]
        
        #self.support_vectors = X[supportIndices]
        #self.support_Y = label[supportIndices]
        #self.alpha = self.beta[supportIndices]  
        #alpha : all_alpha sans les alpha < eps        
        #print('We keep ', len(self.alpha), 'support vectors out of',len(self.all_alpha),'vectors')
        
        #Pour prédire avec classe estimator:
        #self.X_train = X
        self.alpha = self.beta*label
        self.alpha[supportIndices] = np.zeros(n)[supportIndices]
        
        #Bias
        self.b = 0
        for i in range(n):
            self.b = label[i]
            self.b -= np.sum( self.alpha * K[i, :])
        self.b /= n

    def set_parameter(self,C): 
        self.C = C 












































