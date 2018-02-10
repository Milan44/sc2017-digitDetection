import sys
import cv2
import numpy as np

from sklearn.datasets import fetch_mldata
from sklearn.neighbors import KNeighborsClassifier
from scipy import ndimage
from vector import distance, pnt2line
from matplotlib.pyplot import cm 

from skimage import color
from skimage import exposure
from sklearn import datasets
import imutils

import matplotlib.pyplot as plt


def calcLineCoeff(x1, y1, x2, y2):
    a = np.array([[x1,1],[x2,1]])
    b = np.array([y1,y2])
    [k, n] = np.linalg.solve(a,b)
    
    return [k, n]

def passed(k,n,element,x1,x2):
    if(element['pass']==False):
        if(element['center'][1]-k*element['center'][0]-n==0  and x1<center[0]<x2):
            return True
    else:
        return False   


def separetaLine(videoName):
    cap = cv2.VideoCapture(videoName)
    ret, frame = cap.read()

    greenImg = frame.copy()
    blueImg = frame.copy()

    greenImg[:, :, 0] = 0
    greenImg[:, :, 2] = 0

    blueImg[:, :, 1] = 0  
    blueImg[:, :, 2] = 0

    grayGreen = cv2.cvtColor(greenImg, cv2.COLOR_BGR2GRAY)
    grayBlue = cv2.cvtColor(blueImg, cv2.COLOR_BGR2GRAY)

    edgesGreen = cv2.Canny(grayGreen, 50, 150, apertureSize = 3)
    linesGreen = cv2.HoughLinesP(edgesGreen, rho=1, theta=np.pi / 180, threshold=100, minLineLength=100, maxLineGap=10)
    
    edgesBlue = cv2.Canny(grayBlue, 50, 150, apertureSize = 3)
    linesBlue = cv2.HoughLinesP(edgesBlue, rho=1, theta=np.pi / 180, threshold=100, minLineLength=50, maxLineGap=10)

    return linesGreen,linesBlue

    
def findLineCoordinates(linesGreen,linesBlue):
    Gx1 = 0
    Gy1 = 0
    Gx2 = 0
    Gy2 = 0
    for x1, y1, x2, y2 in linesGreen[0]:
        Gx1=x1
        Gy1=y1
        Gx2=x2
        Gy2=y2
    
    for line in linesGreen :
        for x1, y1, x2, y2 in line:
            if x1<Gx1:
                Gx1=x1
                Gy1=y1
            if x2>Gx2:
                Gy2=y2
                Gx2=x2

    Bx1 = 0
    By1 = 0
    Bx2 = 0
    By2 = 0
    for x1, y1, x2, y2 in linesBlue[0]:
        Bx1=x1
        By1=y1
        Bx2=x2
        By2=y2

    for line in linesBlue:
        for x1, y1, x2, y2 in line:
            if x1<Bx1:
                Bx1=x1
                By1=y1
            if x2>Bx2:
                By2=y2
                Bx2=x2

    return Gx1,Gx2,Bx1,Bx2,Gy1,Gy2,By1,By2


cc = -1
def nextId():
    global cc
    cc += 1
    return cc

def inRange(r, item, items):        # inRange fja proverava da li ima brojeva u okolini trenutnog broja
    retVal = []
    for obj in items:
        mdist = distance(item['center'], obj['center'])
        if(mdist<r):
            retVal.append(obj)
    return retVal


def findPassedElements(videoName,Blueline,Greenline):
    elements = []
    t = 0
    counterBlue = 0
    counterGreen = 0
    blueElements = []
    greenElements = []

    kernel = np.ones((2,2), np.uint8)

    cap = cv2.VideoCapture(videoName)
    while (1) :
        ret, img = cap.read()
        if ret == False:
            break
        
        img0 = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        ret, img0 = cv2.threshold(img0, 168, 255, cv2.THRESH_BINARY) 

        img0 = cv2.dilate(img0,kernel)  # radimo 2 puta za redom dilataciju jer su neki brojevi tanje ispisani,
        img0 = cv2.dilate(img0,kernel)  # pa ih ne registruje kada prodju preko linije

        labeled, nr_objects = ndimage.label(img0)
        objects = ndimage.find_objects(labeled)
        
        for i in range(nr_objects):
            loc = objects[i]
            (xc,yc) = ((loc[1].stop + loc[1].start)/2,
                    (loc[0].stop + loc[0].start)/2)
            (dxc,dyc) = ((loc[1].stop - loc[1].start),
                    (loc[0].stop - loc[0].start))
            
            if(dxc>11 or dyc>11):
                cv2.circle(img, (xc,yc), 16, (25, 25, 255), 1)
                elem = {'center':(xc,yc), 'size':(dxc,dyc), 't':t}
                # proverava da li ima jos elemenata u okolini trenutnog elementa
                lst = inRange(20, elem, elements)         # vraca listu nadjenih elemenata u okolini
                nn = len(lst)
                if nn == 0:                 # ako je lista prazna, znaci da je trenutni element novi, pa ga dodajemo
                    elem['id'] = nextId()
                    elem['t'] = t
                    elem['passBlue'] = False
                    elem['passGreen'] = False
                    elem['history'] = [{'center':(xc,yc), 'size':(dxc,dyc), 't':t}]
                    elem['future'] = [] 
                    img1 = cv2.erode(img0,kernel)
                    #img1 = cv2.erode(img1,kernel)
                    contour = img1[yc-dyc/2 : yc+dyc/2, xc-dxc/2 : xc+dxc/2]
                    elem['img'] = cv2.resize(contour, (28, 28))         # cuvam za svaki element njegovu slicicu
                    elements.append(elem)
                elif nn == 1:               # ako u listi ima 1 element, znaci ta je to trenutni element iz proslog frejma
                    lst[0]['center'] = elem['center']       # pa mu menjamo koordinate ventra, broj frejma itd..
                    lst[0]['t'] = t
                    lst[0]['history'].append({'center':(xc,yc), 'size':(dxc,dyc), 't':t}) 
                    lst[0]['future'] = [] 

        for el in elements:
            tt = t - el['t']
            if(tt<3):
                dist, pnt, r = pnt2line(el['center'], Blueline[0], Blueline[1])
                if r>0:
                    cv2.line(img, pnt, el['center'], (0, 255, 25), 1)
                    c = (25, 25, 255)
                    if(dist<9):
                        c = (0, 255, 160)
                        if el['passBlue'] == False:
                            el['passBlue'] = True
                            counterBlue += 1
                            blueElements.append(el)

                dist2, pnt2, r2 = pnt2line(el['center'], Greenline[0], Greenline[1])
                if r2>0:
                    cv2.line(img, pnt2, el['center'], (0, 255, 25), 1)
                    c = (25, 25, 255)
                    if(dist2<9):
                        c = (0, 255, 160)
                        if el['passGreen'] == False:
                            el['passGreen'] = True
                            counterGreen += 1
                            greenElements.append(el)
                
                c = (25, 25, 255)
                cv2.circle(img, el['center'], 16, c, 2)

                id = el['id']
                cv2.putText(img, str(el['id']), 
                (el['center'][0]+10, el['center'][1]+10), cv2.FONT_HERSHEY_SIMPLEX, 1, 255)
                for hist in el['history']:
                    ttt = t-hist['t']
                    if(ttt<100):
                        cv2.circle(img, hist['center'], 1, (0, 255, 255), 1)

                for fu in el['future']:
                    ttt = fu[0]-t
                    if(ttt<100):
                        cv2.circle(img, (fu[1], fu[2]), 1, (255, 255, 0), 1)

        cv2.putText(img, 'CounterBlue: '+str(counterBlue), (380, 90), cv2.FONT_HERSHEY_SIMPLEX, 1,(90,90,255),2)
        cv2.putText(img, 'CounterGreen: '+str(counterGreen), (5, 90), cv2.FONT_HERSHEY_SIMPLEX, 1,(90,90,255),2)

        t += 1

        cv2.imshow('video', img)
        k = cv2.waitKey(1) & 0xff
        if k == 27:
            break
       
    cap.release()
    cv2.destroyAllWindows()

    return blueElements,greenElements
    

def editData(mnist_data):
    for i in range(len(mnist_data)):       
        mnist_img = mnist_data[i].reshape(28,28)
        mnist_img = (mnist_img).astype('uint8')
        mnist_img = exposure.rescale_intensity(mnist_img, out_range=(0, 255))
        
        #kernel = np.ones((2,2), np.uint8)
        #mnist_img = cv2.dilate(mnist_img,kernel) 

        im2, contours, hierarchy = cv2.findContours(mnist_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cords = cv2.boundingRect(contours[0])
        mnist_img_cropped = mnist_img[cords[1] : cords[1] + cords[3], cords[0] : cords[0] + cords[2]]

        mnist_img_resized = cv2.resize(mnist_img_cropped, (28, 28)) 
        """
        cv2.imshow('prvobitna slicica', mnist_img)
        cv2.imshow('cropovana slicica', mnist_img_cropped)
        cv2.imshow('resizovana slicica', mnist_img_resized)
        cv2.waitKey(0)   
        """
        temp = mnist_img_resized.flatten()
        mnist_data[i] = temp  
        
    return mnist_data


def showPoints(videoName, Gx1,Gx2,Bx1,Bx2,Gy1,Gy2,By1,By2):
    [k1, n1] = calcLineCoeff(Gx1, Gy1, Gx2, Gy2)
    [k2, n2] = calcLineCoeff(Bx1, By1, Bx2, By2)
    print "Zelena linina: \n k = %f \n n = %f" %(k1,n1)
    print "Plava linina: \n k = %f \n n = %f" %(k2,n2)
    cap = cv2.VideoCapture(videoName)
    ret, img = cap.read()
    cv2.circle(img,(Gx1,Gy1), 7, (0,0,255), 2)
    cv2.circle(img,(Gx2,Gy2), 7, (0,0,255), 2)
    cv2.circle(img,(Bx1,By1), 7, (0,0,255), 2)
    cv2.circle(img,(Bx2,By2), 7, (0,0,255), 2)
    cv2.imshow('Lines', img)
    cv2.waitKey(0)


mnist = fetch_mldata('MNIST original')
data   = editData(mnist.data) 
labels = mnist.target.astype('int')

knn = KNeighborsClassifier(n_neighbors=1, algorithm='brute').fit(data, labels)

sum = 0
"""
f= open("out.txt","w+")
f.write("RA 139/2014 Milan Suvajdzic\n")
f.write("file	sum\n")
f.close() 
"""
for i in range (10):  
    videoName = 'videos//video-'+str(i)+'.avi'

    linesGreen,linesBlue = separetaLine(videoName)

    Gx1,Gx2,Bx1,Bx2,Gy1,Gy2,By1,By2 = findLineCoordinates(linesGreen,linesBlue)

    #print "Koordinate zelene linije: (%d, %d) i (%d, %d)" %(Gx1,Gy1,Gx2,Gy2)
    #print "Koordinate plave linije: (%d, %d) i (%d, %d)" %(Bx1,By1,Bx2,By2)

    #showPoints(videoName, Gx1,Gx2,Bx1,Bx2,Gy1,Gy2,By1,By2)

    greenLine = [(Gx1,Gy1), (Gx2, Gy2)]
    blueLine = [(Bx1,By1), (Bx2, By2)]

    passedBlueElements,passedGreenElements = findPassedElements(videoName,blueLine,greenLine)

    #print "\nBrojevi kroz plavu liniju: "
    for el in passedBlueElements:
        #cv2.imshow('element', el['img'])
        #cv2.waitKey(0)
        broj = int(knn.predict(el['img'].reshape(1, 784)))
        sum = sum + broj
        #print broj

    #print "Brojevi kroz zelenu liniju: "
    for el in passedGreenElements:
        #cv2.imshow('element', el['img'])
        #cv2.waitKey(0)
        broj = int(knn.predict(el['img'].reshape(1, 784)))
        sum = sum - broj
        #print broj

    print "Suma za video %d je: %d " %(i,sum)
    """
    f= open("out.txt","a+")
    f.write('video-' + str(i) + '.avi ' + str(sum) + '\n')
    f.close()
    """
    sum = 0



