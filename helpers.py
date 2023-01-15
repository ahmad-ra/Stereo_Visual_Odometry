import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R
from math import cos, sin



def circulatMatchingOpticalFlow(ImT1_L,ImT1_R,ImT2_L,ImT2_R, lk_params, featuresOfImT1_L, distanceThreshold, removeWeakMatches=True ):

    p1, st1, err1 = cv.calcOpticalFlowPyrLK(ImT1_L, ImT1_R, featuresOfImT1_L, None, **lk_params)
    p2, st2, err2 = cv.calcOpticalFlowPyrLK(ImT1_R, ImT2_R, p1, None, **lk_params)
    p3, st3, err3 = cv.calcOpticalFlowPyrLK(ImT2_R, ImT2_L, p2, None, **lk_params)
    p4, st4, err4 = cv.calcOpticalFlowPyrLK(ImT2_L, ImT1_L, p3, None, **lk_params)


    d=np.linalg.norm((featuresOfImT1_L-p4).max(1),axis=1)<distanceThreshold

    stereoConstraint= ( ( np.abs(p1[:,:,1]-featuresOfImT1_L[:,:,1]) )<=5).reshape((-1,1))

    st=(st1==1) & (st2==1) & (st3==1) & (st4==1) & d.reshape((-1,1)) & stereoConstraint

    if removeWeakMatches:
        return featuresOfImT1_L[st], p1[st],p2[st],p3[st],p4[st],np.linalg.norm((featuresOfImT1_L-p4).max(1),axis=1)
    
    # return featuresOfImT1_L, p1,p2,p3,p4,st,1




def getStrongFeatures(ImT1_L,ImT1_R,ImT2_L,ImT2_R, lk_params, harrisFeatureParams, distanceThreshold, featureEngine, useCorners, bucketingH, bucketingW):
    '''
    return p0,p1,p2,p3: 4 sets of keypoints that represent strong circular-matched features
    also returns p4 which corresponds to p0 features after tracking them on the full loop of the 4 images, and the tracking error
    '''

    if not useCorners:

        H, W = ImT1_L.shape
        kp = []
        p0=[]
        for y in range(0, H, bucketingH):
            for x in range(0, W, bucketingW):
                imPatch = ImT1_L[y:y + bucketingH, x:x + bucketingW]
                keypoints = featureEngine.detect(imPatch) 
                for pt in keypoints:
                    pt.pt = (pt.pt[0] + x, pt.pt[1] + y)

                if (len(keypoints) > 10):
                    keypoints = sorted(keypoints, key=lambda x: -x.response)
                    for kpt in keypoints[0:10]:
                        kp.append(kpt)
                else:
                    for kpt in keypoints:
                        kp.append(kpt)

        

        p0 = cv.KeyPoint_convert(kp)
        p0 = np.expand_dims(p0, axis=1)

    else:

        p0 = cv.goodFeaturesToTrack(ImT1_L, mask=None, **harrisFeatureParams)

    return circulatMatchingOpticalFlow(ImT1_L,ImT1_R,ImT2_L,ImT2_R, lk_params, p0, distanceThreshold )





def generate3DPoints(points2D_L, points2D_R, Proj1, Proj2):
    numPoints = points2D_L.shape[0]
    d3dPoints = np.ones((numPoints,3))

    for i in range(numPoints):
        pLeft = points2D_L[i,:]
        pRight = points2D_R[i,:]

        X = np.zeros((4,4))
        X[0,:] = pLeft[0] * Proj1[2,:] - Proj1[0,:]
        X[1,:] = pLeft[1] * Proj1[2,:] - Proj1[1,:]
        X[2,:] = pRight[0] * Proj2[2,:] - Proj2[0,:]
        X[3,:] = pRight[1] * Proj2[2,:] - Proj2[1,:]

        [u,s,v] = np.linalg.svd(X)
        v = v.transpose()
        vSmall = v[:,-1]
        vSmall /= vSmall[-1]

        d3dPoints[i, :] = vSmall[0:-1]

    return d3dPoints


def genEulerZXZMatrix(psi, theta, sigma):
    # ref http://www.u.arizona.edu/~pen/ame553/Notes/Lesson%2008-A.pdf
    c1 = cos(psi)
    s1 = sin(psi)
    c2 = cos(theta)
    s2 = sin(theta)
    c3 = cos(sigma)
    s3 = sin(sigma)

    mat = np.zeros((3,3))

    mat[0,0] = (c1 * c3) - (s1 * c2 * s3)
    mat[0,1] = (-c1 * s3) - (s1 * c2 * c3)
    mat[0,2] = (s1 * s2)

    mat[1,0] = (s1 * c3) + (c1 * c2 * s3)
    mat[1,1] = (-s1 * s3) + (c1 * c2 * c3)
    mat[1,2] = (-c1 * s2)

    mat[2,0] = (s2 * s3)
    mat[2,1] = (s2 * c3)
    mat[2,2] = c2

    return mat


def minimizeReprojection(dof,d2dPoints1, d2dPoints2, d3dPoints1, d3dPoints2, w2cMatrix):
    Rmat = genEulerZXZMatrix(dof[0], dof[1], dof[2])
    # Rmat = R.from_euler('XYZ',dof[0:3]).as_matrix()
    
    translationArray = np.array([[dof[3]], [dof[4]], [dof[5]]])
    
    # Rmat = R.from_euler('XZ',dof[0:2]).as_matrix()
    # translationArray = np.array([[dof[2]], [dof[3]], [dof[4]]]) 
    
    temp = np.hstack((Rmat, translationArray))
    perspectiveProj = np.vstack((temp, [0, 0, 0, 1]))

    numPoints = d2dPoints1.shape[0]
    errorA = np.zeros((numPoints,3))
    errorB = np.zeros((numPoints,3))

    forwardProjection = np.matmul(w2cMatrix, perspectiveProj)
    backwardProjection = np.matmul(w2cMatrix, np.linalg.inv(perspectiveProj))
    for i in range(numPoints):
        Ja = np.ones((3))
        Jb = np.ones((3))
        Wa = np.ones((4))
        Wb = np.ones((4))

        Ja[0:2] = d2dPoints1[i,:]
        Jb[0:2] = d2dPoints2[i,:]
        Wa[0:3] = d3dPoints1[i,:]
        Wb[0:3] = d3dPoints2[i,:]

        JaPred = np.matmul(forwardProjection, Wb)
        JaPred /= JaPred[-1]
        e1 = Ja - JaPred

        JbPred = np.matmul(backwardProjection, Wa)
        JbPred /= JbPred[-1]
        e2 = Jb - JbPred

        errorA[i,:] = e1
        errorB[i,:] = e2

    residual = np.vstack((errorA,errorB))
    return residual.flatten()
