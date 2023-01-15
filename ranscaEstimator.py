from helpers import *
from sklearn.base import BaseEstimator



class estimator(BaseEstimator):

    def __init__(self,proj):
        

        self.proj      =proj      


    def fit(self,points3d2dT1,points3d2dT2):
        '''
        shape n x 5
        '''

        dSeed = np.zeros(6)
        optRes = least_squares(minimizeReprojection, dSeed, method='lm', max_nfev=200,
                                                args=( points3d2dT1[:,3:], points3d2dT2[:,3:], points3d2dT1[:,:3] , points3d2dT2[:,:3],self.proj))

        self.dOut_ = optRes.x
     
        return self
    
    def score(self,points3d2dT1,points3d2dT2):

        
        error = minimizeReprojection(self.dOut_, points3d2dT1[:,3:], points3d2dT2[:,3:], points3d2dT1[:,:3] , points3d2dT2[:,:3],self.proj)

        error=1/np.average(error)

        return error


    def predict(self,points3d2dT1):


        dOut=self.dOut_
        Rmat_=genEulerZXZMatrix(dOut[0], dOut[1], dOut[2])
        translationArray_ = np.array([[dOut[3]], [dOut[4]], [dOut[5]]]) 
        
        temp = np.hstack((Rmat_, translationArray_))
        perspectiveProj_ = np.vstack((temp, [0, 0, 0, 1]))
        

        p3dTmp= np.ones((4,points3d2dT1.shape[0]))

        p3dTmp[:3,:]= points3d2dT1[:,:3].T

        #shape 4xn
        p3dTmp2 = np.matmul(np.linalg.inv(perspectiveProj_), p3dTmp )
        p3dTmp2= p3dTmp2/ p3dTmp2[3,:]

        #shape 3 x n
        p2dTmp2= np.matmul( self.proj,p3dTmp2 )
        p2dTmp2= p2dTmp2/ p2dTmp2[2,:]


        points3d2dT2=np.zeros_like(points3d2dT1)

        points3d2dT2[:,:3]=   p3dTmp2[:3,:].T
        points3d2dT2[:,3:]=   p2dTmp2[:2,:].T  

        return points3d2dT2

    def estimated(self):
        return self.dOut_

