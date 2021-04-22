import numpy as np
import math
import cv2
import random as r


def dist(x1,x2,y1,y2):
    return math.sqrt((x1-x2)**2+(y1-y2)**2)

def CalcM(H,j,i):
    return [(H[0][0]*j+H[0][1]*i+H[0][2])/(H[2][0]*j+H[2][1]*i+H[2][2]),
            (H[1][0]*j+H[1][1]*i+H[1][2])/(H[2][0]*j+H[2][1]*i+H[2][2]),
            1]

def spherWght(image):
    h = len(image)
    w = len(image[0])
    hype = ((((h-1)/2)**2)+(((w-1)/2)**2))**.5
    wght = np.zeros([h,w,1], np.float32)
    for i in range(h):
        for j in range(w):
            dist = (((abs(i)-((h-1)/2))**2)+((abs(j)-((w-1)/2))**2))**.5
            wght[i][j] = [1 - dist/hype]
    return wght


def gauss():
    m = [[0.0 for j in range(3)] for i in range(3)]
    

    for i in range(3):
        for j in range(3):
            m[i][j] = g(i+1-3,j+1-3)
    return m


def g(x,y):
    thea = 1.5
    return (math.e**-(((x**2)+(y**2))/(2.0*(thea**2))))/\
           (2.0*math.pi*(thea**2))

def Sobel(matrix, sobel = None):
    if sobel is None:
        sobel = [[0.011955246486766671, 0.023285640551474758, 0.02908024586668896, 0.023285640551474758, 0.011955246486766671],
[0.023285640551474758, 0.04535423476987057, 0.05664058479678963, 0.04535423476987057, 0.023285640551474758],
[0.02908024586668896, 0.05664058479678963, 0.0707355302630646, 0.05664058479678963, 0.02908024586668896],
[0.023285640551474758, 0.04535423476987057, 0.05664058479678963, 0.04535423476987057, 0.023285640551474758],
[0.011955246486766671, 0.023285640551474758, 0.02908024586668896, 0.023285640551474758, 0.011955246486766671],
]
        #sobel = [[1/9 for i in range(3)] for i in range(3)]

    h = (len(matrix)-(len(sobel)-1))
    w = (len(matrix[0])-(len(sobel[0])-1))
    
    out = np.zeros((h,w,1),np.float32)
    
    for i in range(h):
        for j in range(w):
            sum = 0
            for ki in range(len(sobel)):
                for kj in range(len(sobel[0])) :
                    sum+= matrix[i+ki][j+kj] * sobel[ki][kj]
                    
            out[i][j] = sum
                    
    return out 

def makeH(img1,img2):
    # Size 
    imH = len(img1)
    imW = len(img1[0])

    coords = []
    A = np.zeros([8,9], np.float32)

    # Initiate SIFT detector
    sift = cv2.xfeatures2d.SIFT_create()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)

    # BFMatcher with default params
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1,des2, k=2)

    # Apply ratio test
    good = []
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            good.append([m])

    counter = 0
    
    #### Extra Syntax
    #### Get coordinates for the ith match
    minError = -1
    bestH = 0
    for k in range(1000):
        t = []
        for asd in range(4):
            t.append(r.randrange(len(good)))

        counter = 0 
        err = 0
        for i in t:
            qIdx = good[i][0].queryIdx
            tIdx = good[i][0].trainIdx
            x1 = kp1[qIdx].pt[0]
            y1 = kp1[qIdx].pt[1]
            x2 = kp2[tIdx].pt[0]
            y2 = kp2[tIdx].pt[1]
            coords.append(((x1,y1,1),(x2,y2,1)))   

            A[counter] = (0,0,0,-x1,-y1,-1,x1*y2,y2*y1,y2)
            counter +=1

            A[counter] = (x1,y1,1,0,0,0,-x2*x1,-x2*y1,-x2)
            counter+=1

        ## Run SVD
        U, s, V = np.linalg.svd(A, full_matrices=True)
        hCol    = np.zeros((9,1), np.float64)
        hCol    = V[8,:]

        H = hCol.reshape((3,3))
        H = (1/H[2][2])*H
        
        # RANSAC
        for j in range(len(good)):
            qIdx = good[j][0].queryIdx
            tIdx = good[j][0].trainIdx
            x1 = kp1[qIdx].pt[0]
            y1 = kp1[qIdx].pt[1]
            x2 = kp2[tIdx].pt[0]
            y2 = kp2[tIdx].pt[1]

            c = CalcM(H,x1,y1)
            
            err += dist(c[0],x2,c[1],y2)

        if err < minError or minError == -1:
            bestH = H
            minError = err
    return bestH


def main():
    #file2 = "CROPwaterfall1.jpg"
    #file1 = "CROPwaterfall2.jpg"
    
    file1 ="florenceLeft.JPG"
    file2 ="florenceRight.JPG"
    file1 ="mountainLeft.jpg"
    file2 ="mountainRight.jpg"
    file2 ="queenstownLeft.JPG"
    file1 ="queenstownRIght.JPG"

    # For copying color picture
    colorImage1 = cv2.imread(file1)
    colorImage2 = cv2.imread(file2)

    img1 = cv2.imread(file1,0)
    img2 = cv2.imread(file2,0)


    #print("Smallest Error: ", minError)

    H  = makeH(img1,img2)
    print("Homography matrix:\n",H)

    ## Resize Image ##
    xneg = 0
    xpos = len(img2[0])
    yneg = 0
    ypos = len(img2)
    
    topleft  = CalcM(H,0,0)
    topright = CalcM(H,len(img1[0]),0) 
    botleft  = CalcM(H,0,len(img1)) 
    botright = CalcM(H,len(img1[0]),len(img1))

    # Find new top y (neg)
    if topleft[1] < 0 or topright[1] < 0:
        if topleft[1]<topright[1]:
            yneg = topleft[1]
        else:
            yneg = topright[1]

    # Find new left x (neg)
    if topleft[0] < 0 or botleft[0] < 0:
        if topleft[0]<botleft[0]:
            xneg = topleft[0]
        else:
            xneg = botleft[0]

    ypos -= yneg
    xpos -= xneg
    # Find new bot y (large pos)
    if botleft[1]-yneg > ypos or botright[1]-yneg >ypos:  
        if botleft[1]-yneg < botright[1]-yneg:
            ypos = botright[1]-yneg
        else:
            ypos = botleft[1]-yneg

    # Find new right x (large pos)
    if topright[0]-(xneg)>xpos or botright[0]-(xneg)>xpos:
        if topright[0]-xneg<botright[0]-(xneg):
            xpos = botright[0]-xneg
        else:
            xpos = topright[0]-xneg
    
    # Round the right direction
    xneg = math.floor(xneg)
    xpos = math.ceil(xpos)
    yneg = math.floor(yneg)
    ypos = math.ceil(ypos)
    
    #print(xneg,yneg,xpos,ypos)

    # Make new images that will fit everything
    newIm  = np.zeros((ypos,xpos,3), np.float32)
    newIm2 = np.zeros((ypos,xpos,3), np.float32)
    diffSum   = [0,0,0]
    diffCount = 0


    for i in range(len(img2)):
        for j in range(len(img2[0])):
            newIm[i-yneg][j-xneg]=colorImage2[i][j]



########################
########################
########################
########################


    print("[**] Making blending map")
    # spherWght
    img1Sph = spherWght(img1)
    img2Sph = spherWght(img2)

    invH = np.linalg.inv(H)

    newImg1Sph = np.zeros((ypos,xpos,2), np.int)
    newImg2Sph = np.zeros((ypos,xpos,2), np.int)


    wgh1 = np.zeros((ypos,xpos,1), np.float32)
    wgh2 = np.zeros((ypos,xpos,1), np.float32)

    newImg1Sph.fill(-1)
    newImg2Sph.fill(-1)
    
    
    # binaryweights
    bw1 = np.zeros((len(img1),len(img1[0]),1), np.float32)
    bw2 = np.zeros((len(img2),len(img2[0]),1), np.float32)

    bw1.fill(1)
    bw2.fill(1)

    # IM2 MAP
    for i in range(len(img2)):
        for j in range(len(img2[0])):
            newImg2Sph[i-yneg][j-xneg]=[i,j]

    # IM1 MAP
    for i in range(yneg,len(newIm)):
        for j in range(xneg,len(newIm[0])):
            newx = (invH[0][0]*j+invH[0][1]*i+invH[0][2])/(invH[2][0]*j+invH[2][1]*i+invH[2][2])
            newy = (invH[1][0]*j+invH[1][1]*i+invH[1][2])/(invH[2][0]*j+invH[2][1]*i+invH[2][2])
            try:
                if 0 < newy and 0 < newx:
                    img1Sph[int(newy)][int(newx)]
                    newImg1Sph[int(round(i-yneg))][int(round(j-xneg))] = [int(newy),int(newx)]
                    #tesbs[int(round(i-yneg))][int(round(j-xneg))] = img1Sph[int(newy)][int(newx)]
                    #print(newImg1Sph[int(round(i-yneg))][int(round(j-xneg))])#########
            except IndexError:
                pass

    
    # Make binary weights
    #img1Sph
    #img2Sph

    for i in range(len(newIm)):
        for j in range(len(newIm[0])):
            y1,x1=newImg1Sph[i][j]
            y2,x2=newImg2Sph[i][j]  

            if x1 != -1 and x2 != -1:

                try:
                   if img1Sph[y1][x1] < img2Sph[y2][x2]:
                        bw1[y1][x1] = 0
                   else:
                        bw2[y2][x2] = 0
                except IndexError:
                    pass
    

    # wgh1 MAP
    for i in range(yneg,len(newIm)):
        for j in range(xneg,len(newIm[0])):
            newx = (invH[0][0]*j+invH[0][1]*i+invH[0][2])/(invH[2][0]*j+invH[2][1]*i+invH[2][2])
            newy = (invH[1][0]*j+invH[1][1]*i+invH[1][2])/(invH[2][0]*j+invH[2][1]*i+invH[2][2])
            try:
                if 0 < newy and 0 < newx:
                    img1Sph[int(newy)][int(newx)]
                    wgh1[int(round(i-yneg))][int(round(j-xneg))] = bw1[int(newy)][int(newx)]

            except IndexError:
                pass

    # IM2 MAP
    for i in range(len(img2)):
        for j in range(len(img2[0])):
            wgh2[i-yneg][j-xneg]=bw2[i][j]


    wgh2a = Sobel(wgh2)
    wgh2b = Sobel(wgh2a)
    wgh2c = Sobel(wgh2b)

    wgh1a = Sobel(wgh1)
    wgh1b = Sobel(wgh1a)
    wgh1c = Sobel(wgh1b)

    for i in range(len(img2)):
        for j in range(len(img2[0])):
            wgh2[i-yneg][j-xneg]=bw2[i][j]
    '''
    cv2.imshow("a1" ,wgh2a)
    cv2.imshow("b1" ,wgh2b)
    cv2.imshow("c1" ,wgh2c)
    cv2.imshow("11" ,wgh2)

    cv2.imshow("a2" ,wgh1a)
    cv2.imshow("b2" ,wgh1b)
    cv2.imshow("c2" ,wgh1c)
    cv2.imshow("2" ,wgh1)

    cv2.imshow("sph1" ,img1Sph)
    cv2.imshow("sph2" ,img2Sph)
    cv2.waitKey(0)
    cv2.destroyAllWindows()'''

########################
########################
########################
########################



    print("[**] Making transformed picture")
    H = np.linalg.inv(H)
    for i in range(yneg,len(newIm)):
        for j in range(xneg,len(newIm[0])):
            newx = (H[0][0]*j+H[0][1]*i+H[0][2])/(H[2][0]*j+H[2][1]*i+H[2][2])
            newy = (H[1][0]*j+H[1][1]*i+H[1][2])/(H[2][0]*j+H[2][1]*i+H[2][2])
            try:
                if 0 < newy and 0 < newx:
                    newIm2[int(round(i-yneg))][int(round(j-xneg))] =\
                    colorImage1[int(newy)][int(newx)]
            except IndexError:
                pass
                
    # Adjust brightness
    print("[**] Finding lighting correction")
    for i in range(len(newIm)):
        for j in range(len(newIm[0])):
            if sum(newIm[i][j]) != 0 and sum(newIm2[i][j]) != 0:
                for k in range(3):
                    diffSum[k] += newIm[i][j][k] - newIm2[i][j][k]
                    diffCount  += 1
                    
    diffSum = [i/diffCount for i in diffSum]
    
    print("Correction:",diffSum)
    
    for i in range(yneg,len(newIm)):
        for j in range(xneg,len(newIm[0])):
            newx = (H[0][0]*j+H[0][1]*i+H[0][2])/(H[2][0]*j+H[2][1]*i+H[2][2])
            newy = (H[1][0]*j+H[1][1]*i+H[1][2])/(H[2][0]*j+H[2][1]*i+H[2][2])
            try:
                if 0 < newy and 0 < newx:
                    newIm[int(round(i-yneg))][int(round(j-xneg))] =\
                    colorImage1[int(newy)][int(newx)]+diffSum
            except IndexError:
                pass
        
    cv2.imshow("OUT_PICs",newIm2/255)
    cv2.imshow("OUT_PIC" ,newIm /255)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



if __name__ == "__main__":
    main()



