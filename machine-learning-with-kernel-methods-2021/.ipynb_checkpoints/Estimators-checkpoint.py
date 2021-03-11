import numpy as np
import pandas as pd
import numpy.linalg 
from tqdm import tqdm
import numba
from numba import njit,vectorize, jit
import time
import scipy


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




class estimator(): 
    def __init__(self , Kernel): 
        self.Kernel = Kernel
        self.Kernel_train = None 
        self.alpha = None 
        
    def predict_proba(self,Kernel_test): 
        if (self.alpha == None).any()==True  : 
            print("Il faut d'abord fitter les données")
        else : 
            return  sigmoid(self.alpha@Kernel_test)
    
    def predict(self,K_test): 
        if (self.alpha == None).any()==True : 
            print("Il faut d'abord fitter les données")
        else : 
            prob = self.predict_proba(K_test)
            return prob>0.5
    def cross_val(self, Xtrain,ytrain,cv): 
        mistake = 0
        Xtrainsplit,Xvalsplit,ytrainsplit,yvalsplit = cross_val_split(Xtrain,ytrain,cv)
        for xtrain,xval,ytrain,yval in tqdm(zip(Xtrainsplit,Xvalsplit, ytrainsplit, yvalsplit)):
            kernel_train = to_Kernel_train(xtrain,self.Kernel)
            self.fit(xtrain,ytrain)
            kernel_val = to_Kernel_test(xtrain,xval,self.Kernel)
            pred = self.predict(kernel_val)
            mistake+=np.sum(np.abs(pred-yval))
        print('Pourcentage of errors : ', mistake/Xtrain.shape[0])
        return mistake/Xtrain.shape[0]
    
    
    

#lam = 1e-8 est bien
class KRR(estimator): 
    def __init__(self , Kernel, lam = 1e-8): 
        super(KRR, self).__init__(Kernel)
        self.lam = lam 
        
    def fit(self, K_train, y): 
        y_copy = (2*y-1).copy()
        self.Kernel_train = K_train
        self.alpha = fit_KRR(self.Kernel_train, self.lam, y_copy)















































