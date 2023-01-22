import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

from helpers import *
from ranscaEstimator import estimator
from sklearn import linear_model


import time


###get intirinsics#####

calibFileName='00/calib.txt'
calibFile = open(calibFileName, 'r').readlines()

P1Vals = calibFile[0].split()
Proj1 = np.zeros((3,4))
for row in range(3):
    for column in range(4):
        Proj1[row, column] = float(P1Vals[row*4 + column + 1])

P2Vals = calibFile[1].split()
Proj2 = np.zeros((3,4))
for row in range(3):
    for column in range(4):
        Proj2[row, column] = float(P2Vals[row*4 + column + 1])
###########



######initialization###########

groundTruthTraj = []
poseFile ='00/00.txt'
# poseFile ='02/02.txt'
fpPoseFile = open(poseFile, 'r')
groundTruthTraj = fpPoseFile.readlines()

canvasH = 1200
canvasW = 600
traj = np.zeros((canvasH,canvasW,3), dtype=np.uint8)

translation=None
rotation=None

frm=0

endFrame=4539
startFrame=0

lk_params = dict(winSize=(15, 15),
                maxLevel=2,
                criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

harrisFeatureParams = dict(maxCorners=100,
                    qualityLevel=0.3,
                    minDistance=7,
                    blockSize=7)

distanceThreshold=0.1

keyFrame=1
featureEngine=cv.FastFeatureDetector_create()

useCorners=True
bucketingH=10
bucketingW=20

###########################

savedPoses = open('savedPoses.txt', 'w')

start_time = time.time()
for frm in range(startFrame+1, endFrame+1,keyFrame):
    
    print(frm)
    ImT1_L = cv.imread('00/image_0/'+'{0:06d}'.format(frm-1)+'.png', 0)  # 0 flag returns a grayscale image
    ImT1_R = cv.imread('00/image_1/'+'{0:06d}'.format(frm-1)+'.png', 0)
    ImT2_L = cv.imread('00/image_0/'+'{0:06d}'.format(frm)+'.png', 0)
    ImT2_R = cv.imread('00/image_1/'+'{0:06d}'.format(frm)+'.png', 0)

    p0,p1,p2,p3,p4,errs=getStrongFeatures(ImT1_L,ImT1_R,ImT2_L,ImT2_R, lk_params, harrisFeatureParams, distanceThreshold, featureEngine, useCorners, bucketingH, bucketingW)
       # p0 is T1_L, p1 is T1_R, p2 is T2_R, p3 is T2_L

    if p0.shape[0]<6:
        print('oops! '+ str(frm))
        p0,p1,p2,p3,p4,errs=getStrongFeatures(ImT1_L,ImT1_R,ImT2_L,ImT2_R, lk_params, harrisFeatureParams, distanceThreshold, featureEngine, False, bucketingH, bucketingW)
        

    if p0.shape[0]<6:
        print('skipped frame'+ str(frm)+' min err is '+str(errs.min()))
        continue


    d3dPointsT1 = generate3DPoints(p0, p1, Proj1, Proj2)
    d3dPointsT2 = generate3DPoints(p3, p2, Proj1, Proj2)


    #Ransac, 6 points
    est=estimator(Proj1)
    inlierThreshold=1.0
    r=linear_model.RANSACRegressor(est, 6)

    p00=np.hstack([d3dPointsT1,p0])
    p11=np.hstack([d3dPointsT2,p3])
    r.fit(p00,p11)


    #get parameters of the best ransac model
    dSeed = np.zeros(6)
    optRes = least_squares(minimizeReprojection, dSeed, method='lm', max_nfev=200,
                                            args=(p0[r.inlier_mask_], p3[r.inlier_mask_], d3dPointsT1[r.inlier_mask_], d3dPointsT2[r.inlier_mask_], Proj1))

    #get error in x,y,z, for possible printing of the value
    error = minimizeReprojection(optRes.x, p0, p3, d3dPointsT1, d3dPointsT2, Proj1)

    eCoords = error.reshape((d3dPointsT1.shape[0]*2, 3))

    totalError = np.sum(np.linalg.norm(eCoords, axis=1))

    dOut = optRes.x

    ####get tranlation/rotation values####

    Rmat = genEulerZXZMatrix(dOut[0], dOut[1], dOut[2])
    # Rmat = R.from_euler('XYZ',dOut[0:3]).as_matrix()    
    translationArray = np.array([[dOut[3]], [dOut[4]], [dOut[5]]]) 

    # Rmat = R.from_euler('XZ',dOut[0:2]).as_matrix()
    # translationArray = np.array([[dOut[2]], [dOut[3]], [dOut[4]]]) 


    # lines= Rmat[0,0]+' '+Rmat[0,1]+' '+Rmat[0,2]+' '+Rmat[1,0]+' '+Rmat[1,1]+' '+Rmat[1,2]+' '+Rmat[2,0]+' '+Rmat[2,1]+' '+Rmat[2,2]+' '+translationArray[0,0]+' '+translationArray[1,0]+' '+translationArray[2,0] 

    if (isinstance(translation, np.ndarray)):
        translation = translation + np.matmul(rotation, translationArray)
    else:
        translation = translationArray

    if (isinstance(rotation, np.ndarray)):
        # rotation = np.matmul(Rmat, rotation) orig
        rotation = np.matmul( rotation, Rmat)
    else:
        rotation = Rmat

    line=''
    for i in range(3):
        for j in range(3):
            line= line+str(rotation[i,j])+' '
        line=line+str(translation[i,0])+' '

    # line=line+str(translation[0,0])+' '+str(translation[1,0])+' '+str(translation[2,0])     
    savedPoses.write(line[:-1])
    savedPoses.write('\n')

    #####plotting######
    canvasWCorr = 290
    canvasHCorr = 200
    draw_x, draw_y = int(translation[0])+canvasWCorr, int(translation[2])+canvasHCorr
    grndPose = groundTruthTraj[frm].strip().split()
    grndX = int(float(grndPose[3])) + canvasWCorr
    grndY = int(float(grndPose[11])) + canvasHCorr

    cv.circle(traj, (grndX,grndY), 1, (0,0,255), 2)
    cv.rectangle(traj, (10, 20), (600, 60), (0,0,0), -1)
    text = "Coordinates: x=%2fm y=%2fm z=%2fm"%(translation[0],translation[1],translation[2])
    cv.putText(traj, text, (20,40), cv.FONT_HERSHEY_PLAIN, 1, (255,255,255), 1, 8)
    cv.circle(traj, (draw_x, draw_y), 1, (frm*255/(endFrame-startFrame),255-frm*255/(endFrame-startFrame),0), 1)


    ImT2_L=cv.cvtColor(ImT2_L,cv.COLOR_GRAY2RGB)
    for i, (new, old) in enumerate(zip(p3, p0)):
        a, b = new.ravel()
        c, d = old.ravel()
        ImT2_L = cv.line(ImT2_L, (int(a), int(b)), (int(c), int(d)), (0,255,0) , 5)
        ImT2_L = cv.circle(ImT2_L, (int(c), int(d)), 5, (255,0,0), -1)

    a= np.pad(ImT2_L, [(0,1200-376 ), (0, 0),(0,0)], mode='constant', constant_values=0)

    cv.imshow('Trajectory', np.concatenate((traj, a), axis=1))

    if cv.waitKey(1) == 27: 
            break  # esc to quit


end_time = time.time()

cv.waitKey()
cv.destroyAllWindows()


savedPoses.close()
