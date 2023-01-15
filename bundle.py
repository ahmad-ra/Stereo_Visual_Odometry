#uncompleted

from scipy.sparse import lil_matrix
from helpers import *

from itertools import chain

def getCamIndices(pairsOfImages, lk_params, harrisFeatureParams, distanceThreshold, featureEngine, useCorners, bucketingH, bucketingW,Proj1,Proj2):

    '''
    pairsOfImages: shape nx2
    '''
    pts2dArray=[]
    pts3dArray = []


    im1_L=pairsOfImages[0][0]
    im1_R=pairsOfImages[0][1]
    im2_L=pairsOfImages[1][0]
    im2_R=pairsOfImages[1][1]

    featuresOfIm1_L=  getFeaturesOfImg(im1_L, harrisFeatureParams,  featureEngine, useCorners, bucketingH, bucketingW )

    # print(featuresOfIm1_L.shape)
    p0,p1,p2,p3,p4,indices,_=circulatMatchingOpticalFlow(im1_L,im1_R,im2_L,im2_R, lk_params, featuresOfIm1_L, distanceThreshold, False )
    # indices=indices.astype(int)
    pts2dArray.append(p0)

    # print(p0.shape)
    # print(p1.shape)
    # print(Proj1.shape)
    # print(Proj2.shape)
    pts3d = generate3DPoints(p0.reshape((-1,2)), p1.reshape((-1,2)), Proj1, Proj2)

    pts3dArray.append(pts3d)

    numPoints=len(p0)
    numCams=len(pairsOfImages)
    camsPtsMap = np.zeros((numCams,numPoints))
    indicesArray = np.zeros((numCams,numPoints))

    # print(np.sum(indices))

    # print(indices)

    indicesArray[0,:]= indices.T
    # camPts[0,indices]=1
    # camPts[1,indices]=1

    for i in range(2,len(pairsOfImages),1):

        im1_L=pairsOfImages[i-1][0]
        im1_R=pairsOfImages[i-1][1]
        im2_L=pairsOfImages[i][0]
        im2_R=pairsOfImages[i][1]


        p0,p1,p2,p3,p4,indicesTmp,_=circulatMatchingOpticalFlow(im1_L,im1_R,im2_L,im2_R, lk_params, p3, distanceThreshold, False )
        pts2dArray.append(p0)
        
        pts3d = generate3DPoints(p0.reshape((-1,2)), p1.reshape((-1,2)), Proj1, Proj2)

        # pts3d = generate3DPoints(p0, p1, Proj1, Proj2)

        pts3dArray.append(pts3d)

        # indicesTmp=indicesTmp.astype(int)

        # indices=circulatMatchingOpticalFlow(im1_L,im1_R,im2_L,im2_R, lk_params, p3, distanceThreshold, True ):

        # indices= indices | indicesTmp
        indices= indices&indicesTmp
        indicesArray[i-1,:]= indices.T
        indicesArray[i,:]= indices.T

        # print(np.sum(indices))
        #the output should be p3 without indexing it, and the indices


        # camPts[i,indices]=1
        # camPts[i-1,indices]=1

    pts2dArray.append(p3)
    
    pts3d = generate3DPoints(p3.reshape((-1,2)), p2.reshape((-1,2)), Proj1, Proj2)

    # pts3d = generate3DPoints(p0, p1, Proj1, Proj2)

    pts3dArray.append(pts3d)
    camsPtsMap[indicesArray.astype(bool)]=1
    return camsPtsMap, numPoints,   pts2dArray ,pts3dArray



def bundle_adjustment_sparsity(n_cameras, n_points, camera_indices, point_indices):
    m = camera_indices.size * 2
    n = n_cameras * 9 + n_points * 3
    A = lil_matrix((m, n), dtype=int)

    i = np.arange(camera_indices.size)
    for s in range(9):
        A[2 * i, camera_indices * 9 + s] = 1
        A[2 * i + 1, camera_indices * 9 + s] = 1

    for s in range(3):
        A[2 * i, n_cameras * 9 + point_indices * 3 + s] = 1
        A[2 * i + 1, n_cameras * 9 + point_indices * 3 + s] = 1

    return A



def minimizeReprojectionBundle(dof, d2dPoints2, d3dPoints1,  w2cMatrix):
    Rmat = genEulerZXZMatrix(dof[0], dof[1], dof[2])
    # Rmat = R.from_euler('XYZ',dof[0:3]).as_matrix()
    translationArray = np.array([[dof[3]], [dof[4]], [dof[5]]])
    
    # Rmat = R.from_euler('XZ',dof[0:2]).as_matrix()
    # translationArray = np.array([[dof[2]], [dof[3]], [dof[4]]]) 
    
    temp = np.hstack((Rmat, translationArray))
    perspectiveProj = np.vstack((temp, [0, 0, 0, 1]))
    #print (perspectiveProj)

    numPoints = d2dPoints2.shape[0]
    errorA = np.zeros((numPoints,3))
    errorB = np.zeros((numPoints,3))

    backwardProjection = np.matmul(w2cMatrix, np.linalg.inv(perspectiveProj))
    for i in range(numPoints):
        Jb = np.ones((3))
        Wa = np.ones((4))

        Jb[0:2] = d2dPoints2[i,:]
        Wa[0:3] = d3dPoints1[i,:]



        JbPred = np.matmul(backwardProjection, Wa)
        JbPred /= JbPred[-1]
        e2 = Jb - JbPred

        errorB[i,:] = e2

    residual = errorB
    print(np.sum(residual.flatten()))
    return residual.flatten()


def fun(params, n_cameras, n_points, pts2dArray,  camsPtsMap,w2cMatrix):
    """Compute residuals.
    
    `params` contains camera parameters and 3-D coordinates.
    """
    camera_poses = params[:n_cameras * 6].reshape((n_cameras, 6))
    points_3d = params[n_cameras * 6:].reshape((n_points, 3))
    
    errors=[]
    i:n_cameras

    #array of 2d points, consisting of p0 for all the pairs
    # pts2dArray
    # pts3dArray

    Transformation=np.diag([1,1,1,1])
    for i in range(1,n_cameras,1):

        dof=camera_poses[i]

        Rmat = genEulerZXZMatrix(dof[0], dof[1], dof[2])
        translationArray = np.array([[dof[3]], [dof[4]], [dof[5]]])
        temp = np.hstack((Rmat, translationArray))
        perspectiveProj = np.vstack((temp, [0, 0, 0, 1]))

        #tranf of cam i wrt cam0
        Transformation=Transformation*perspectiveProj

        r,p,yaw=R.from_matrix(Transformation[:3,:3]).as_euler('ZXZ')
        x,y,z=Transformation[:-1,3]

        dof=np.array([x,y,z,r,p,yaw])
        # Rmat = R.from_euler('XZ',dof[0:2]).as_matrix()
        # translationArray = np.array([[dof[2]], [dof[3]], [dof[4]]]) 
        

        pts2dT2 = pts2dArray[i]
        pts2dT2=pts2dT2[camsPtsMap[i].T.astype(np.uint8)]


        pts3dT1 = points_3d
        pts3dT1=pts3dT1[camsPtsMap[i].T.astype(np.uint8)]


        err=minimizeReprojectionBundle(dof, pts2dT2, pts3dT1, w2cMatrix)
        # print(err.shape)
        errors.append(err)


    # list(chain.from_iterable(errors))
    # points_proj = project(points_3d[point_indices], camera_params[camera_indices])
    
    return np.array(errors).ravel()

# res = least_squares(fun, x0, jac_sparsity=A, verbose=2, x_scale='jac', ftol=1e-4, method='trf',
#                     args=(n_cameras, n_points, camera_indices, point_indices, points_2d))



pairsOfImages=[]
# pts2dArray=[]

n_cameras=5
for frm in range(n_cameras):
    
    # print(frm)
    # ImT1_L = cv.imread('image_0/'+'{0:06d}'.format(frm-1)+'.png', 0)  # 0 flag returns a grayscale image
    # ImT1_R = cv.imread('image_1/'+'{0:06d}'.format(frm-1)+'.png', 0)
    
    ImT_L = cv.imread('image_0/'+'{0:06d}'.format(frm)+'.png', 0)
    ImT_R = cv.imread('image_1/'+'{0:06d}'.format(frm)+'.png', 0)

    pairsOfImages.append([ImT_L,ImT_R])



camPtsMap,n_points,pts2dArray,pts3dArray=getCamIndices(pairsOfImages,lk_params,harrisFeatureParams,distanceThreshold,featureEngine,useCorners,bucketingH,bucketingW,Proj1,Proj2)


x0 = np.hstack( [np.array(camera_poses).ravel(), pts3dArray[0].ravel() ] )

a=fun(x0, n_cameras, n_points, pts2dArray,  camPtsMap,Proj1)
 

res = least_squares(fun, x0,  verbose=2,  method='lm',
                    args=(n_cameras, n_points, pts2dArray,  camPtsMap,Proj1))
