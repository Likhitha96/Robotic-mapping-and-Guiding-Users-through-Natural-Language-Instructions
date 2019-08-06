from __future__ import division
import os
import pandas as pd
import csv
import ast
from math import sqrt
import math
import json
import re
import matplotlib.pyplot as plt
import cv2 as cv
from djikstra import navigate
from keywords import output_kw, get_phrase

def preprocessRoi(objData):
    '''
    This performs functions to modify data into readable form
    :param objData: resulting data after object detection
    :return: preprocessed data
    '''
    a=objData['roi']
    roi=[]
    for i in range(len(objData)):
            b=a[i].split("\n")
            con=""
            for item in b[:-1]:
                item=re.sub("\s+", ",", item.strip())
                con = con+item+","
            item = re.sub("\s+", ",", b[len(b)-1].strip())
            con=con+item
            con=con.replace("[,","[")
            con=ast.literal_eval(con)
            roi.append(con)
    return roi

class Image:
    '''
    Image in a viewpoint
    '''
    def __init__(self,index,viewpointId,objects ):
        self.index = index
        self.viewpointId = viewpointId #viewpointID
        self.objects=objects
        self.sortedIndex = []


    def sortByDistances(self):
        '''

        :return: index of objects in the image sorted by distance
        '''
        roi = self.objects['roi']
        roi_index = sorted(range(len(roi)),key = lambda p: sqrt((roi[p][2])**2 + (roi[p][3])**2))
        self.sortedIndex = roi_index
        
    def sortArray(self):
        '''

        :return: objects sorted using sorted index
        '''
        arr=[]
        arr2 = []
        if len(self.sortedIndex)!=0:
            for i in self.sortedIndex:
                arr.append(self.objects['objects'][i])
                arr2.append(self.objects['roi'][i])
                
            self.objects['objects'] = arr
            self.objects['roi'] = arr2
            
class Location:
    '''
    Viewpoint in a scan
    '''
    def __init__(self, lat, lon, viewpoint):
        self.lat = lat
        self.lon = lon
        self.viewpoint = viewpoint
        self.navigableLocations = [] #adjacencyList
        self.imageObjects = []
    
    def readObjects(self):
        '''
        read file containing objects in a viewpoint
        :return:
        '''
        with open('objects/'+self.viewpoint+'.tsv') as f:
            objData = pd.DataFrame.from_csv(f, sep='\t')
            return objData

    def imageForm(self,objData):
        '''
        :param objData: objects in the images within the viewpoint with their class name and roi
        :return: array of images and their objects, roi in the  sorted order of images
        '''
        arr=[]
        roi=preprocessRoi(objData)
        ind = objData.index
        for i in range(len(objData)):
            dp = objData.iloc[i]
            index = ind[i].split('.')[0]
            objDatapoint = {'nob': dp['number_of_objects'],'objects': ast.literal_eval(dp['objects']),'roi':roi[i] }       
            img=Image(index,self.viewpoint,objDatapoint)
            arr.append(img)
            arr.sort(key = lambda p: p.index)
            self.imageObjects = arr
    
    def resolveObjects(self):
        '''
        This functions sorts objects within an image of the viewpoint based on their roi coordinates
        :return: sorted objects
        '''
        for i in range(len(self.imageObjects)):
            self.imageObjects[i].sortByDistances()
            self.imageObjects[i].sortArray()
            
    def resolveObjectsfromAngle(self,start,size):
        '''

        :param start: start image
        :param size: number of images
        :return: objects and rois from start to start+size
        '''
        i=0
        objects=[]
        roi=[]
        while i < size:
            if start+i >= len(self.imageObjects):
                    size = size-i
                    start=0
                    i=0
            #print(start+i)
            for k in range(len(self.imageObjects[start+i].objects['objects'])):
                #print(k)
                objects.append(self.imageObjects[start+i].objects['objects'][k])
                roi.append(self.imageObjects[start+i].objects['roi'][k])
            i=i+1
        return objects,roi
    

    def prospectiveNodes(self,location):
        '''

        :param location: global coordinates in real world
        :return: viewpoints that are less than measurement distance
        '''
        prospective = {}
        for i in location:
            if i != self.viewpoint:
                nLat = location[i].lat
                nLon = location[i].lon
                cPoint = [self.lat,self.lon]
                nPoint = [nLat,nLon]
                #print(cPoint,nPoint)
                dist=findDistance(cPoint,nPoint)
                angle=findAngle(cPoint,nPoint)
                #print(i,dist,angle)
                measurement_dist = 2.5
                if(dist<=measurement_dist):
                    prospective[i] = angle
        return prospective

        
    def sift(self,viewpointId,idx):
        '''
        Computes sift similarity between two images
        :param viewpointId:
        :param idx: image number in the viewpoint
        :return: sift similarity measure
        '''
        vis = cv.imread('img/'+viewpointId+'/'+viewpointId+'_'+str(idx)+'.jpg',cv.IMREAD_GRAYSCALE)
        vis2 = cv.imread('img/'+self.viewpoint+'/'+self.viewpoint+'_'+str(idx)+'.jpg',cv.IMREAD_GRAYSCALE)
        sift = cv.xfeatures2d.SIFT_create()

    # find the keypoints and descriptors with SIFT
        kp1, des1 = sift.detectAndCompute(vis,None)
        kp2, des2 = sift.detectAndCompute(vis2,None)
    # BFMatcher with default params
        bf = cv.BFMatcher()
        if des1 is None or des2 is None:
            return 0
        else:
            matches = bf.knnMatch(des1,des2,k=2)
    # Apply ratio test
        good = []
        for m,n in matches:
            if m.distance < 0.75*n.distance:
                good.append([m])
        return len(good)

def readLocation():
        with open('location.tsv') as f:
            objData = pd.DataFrame.from_csv(f, sep='\t')
            return objData

def createNodes(location):
    '''

    :param location: global coordinates of all the 2D locations
    :return: viewpoints as Location objects
    '''
    viewpoints=location.index
    loc={}
    for i in range(len(viewpoints)):
        point = ast.literal_eval(location.iloc[i]['point'])
        loc[viewpoints[i]]=Location(point[0],point[1],viewpoints[i])
    return loc
        

def findDistance(a,b):
    '''

    :param a: Point 1 (global coordinates of viewpoint 1)
    :param b: Point 2 (global coordinates of viewpoint 2)
    :return: Distance between two points
    '''
    return sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)
    
def findAngle(a,b):
    '''

    :param a: Point 1 (global coordinates of viewpoint 1)
    :param b: Point 2 (global coordinates of viewpoint 2)
    :return: Angle between two viewpoints
    '''
    if (a[1]!=b[1]):
        angle=abs(b[0]-a[0])/abs(b[1]-a[1])
        angle=math.degrees(math.atan(angle))
        if(a[1]<b[1] and a[0]>b[0]):
            angle = 0-angle
        elif(a[1]>b[1] and a[0]>b[0]):
            angle = -180 + angle
        elif(a[1]>b[1] and a[0]<b[0]):
            angle = 180 - angle
        
        #print(angle)
        
        return angle
    
def findReverseAngle(angle):
    '''
    Reverse direction to angle
    :param angle: angle between two viewpoints
    :return:  reverse angle
    '''
    if(angle>=0):
        return -180+angle
    else:
        return 180+angle

def idFromAngle(angle):
    '''

    :param angle: angle between two viewpoints
    :return: return image id from angle
    '''
    if angle>=0 and angle<45:
        return 0
    elif angle>=45 and angle<90:
        return 1
    elif angle>=90 and angle<135:
        return 2
    elif angle>=135 and angle<180:
        return 3
    elif angle>=-180 and angle<-135:
        return 4
    elif angle>=-135 and angle<-90:
        return 5
    elif angle>=-90 and angle<-45:
        return 6
    elif angle>=-45 and angle<0:
        return 7
    
def jaccard_similarity(list1, list2):
    '''

    :param list1: objects from viewpoint 1
    :param list2: objects from viewpoint 2
    :return: intersection of objects / union of objects
    '''
    s1 = set(list1)
    s2 = set(list2)
    den = min(len(s1),len(s2))
    if den!=0:
        return len(s1.intersection(s2)) / den   
    else:
        return 0

        
def final_prospectives(viewpointId,locationList):
    '''
    :param viewpointId:
    :param locationList: list of locations
    :return: viewpoints that have different measures greater than a threshold are final prospectives
    '''
    loc = locationList[viewpointId]
    loc.imageForm(loc.readObjects())
    loc.resolveObjects()
    prospective=loc.prospectiveNodes(locationList)
    #print(prospective)
    #print(len(prospective))
    count=0
    final_prospective=[]
    print(viewpointId)
    for i in prospective:
        loc1 = locationList[i]
        loc1.imageForm(loc1.readObjects())
        loc1.resolveObjects()
        idx=idFromAngle(prospective[i])
        idx2 = idFromAngle(findReverseAngle(prospective[i]))
        a1,_ = loc1.resolveObjectsfromAngle(idx,2)
        a2,_ = loc1.resolveObjectsfromAngle(idx2,2)
        b1,_= loc.resolveObjectsfromAngle(idx,2)
        b2,_= loc.resolveObjectsfromAngle(idx2,2)
        #print(jaccard_similarity(a1,b1))
        #print(jaccard_similarity(a2,b2))
        #print(jaccard_similarity(a2,b1))
        sift_features_idx = loc.sift(i,idx)
        sift_features_ridx = loc.sift(i,idx2)
        #print(sift_features_idx)
        #print(sift_features_ridx)
        if jaccard_similarity(a1,b1)>=0.5 and jaccard_similarity(a2,b2)>=0.5:
            #print("in jac",i)
            final_prospective.append(i)
        elif  sift_features_idx>=20 and sift_features_ridx>=20:
            #print("in sift",i)
            final_prospective.append(i)
    return final_prospective

#get 2d locations in the house
location=readLocation()
#get all viewpoint ids
locationList = createNodes(location)
n_graph = {}
#get reachability graph
for i in locationList.keys():
    n_graph[i] = final_prospectives(i,locationList)
df = []
pred_paths = []

def readScan():
    with open('R2R_train.json') as f:
        objData = pd.read_json(f)
        return objData


data = readScan()

#select one scan
data = data[data['scan'] == '17DRP5sb8fy']
nodes = []
instr = []
gt_paths = []

#gt_paths is ground truth path for a source and destination
#nodes is source node and destination node
for i in range(len(data)):
    paths = data.iloc[i]['path']
    gt_paths.append(paths)
    nodes.append([paths[0], paths[-1]])
    instr.append(data.iloc[i]['instructions'][0])

df = []
pred_paths = []

#for all such nodes, compute path, translate into keywords
for i in range(len(nodes)):
    loc1 = nodes[i][0]
    loc2 = nodes[i][1]

    #compute path
    path=navigate(loc1,loc2,n_graph)
    pred_paths.append(path)

    #translate into keywords
    candidate = output_kw(path,locationList)
    phrase = get_phrase(candidate)
    df.append([phrase,instr[i]])


df = pd.DataFrame(df)
df.columns = ["src","tgt"]
df.to_csv("../data/triple-data/test.csv",sep = ',')