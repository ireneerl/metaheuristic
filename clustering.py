
import sys
import os
import os.path
import time
import math
import csv

def csv_to_numpy(fileName):
    openfile = open(fileName, 'r')
    data = [line.split(',') for line in openfile.readlines()]
    print("csv_to_numpy() finished",fileName)
    # print data
    return data

def multispectral_classification(data, size):
    xoff, yoff = size,size
    n_clust = 10
    outresults=[[0 for x in range(xoff)] for y in range(yoff)]
    kmeans_result = kmeans_scratch(size, n_clust, data)
    result2D = zip(*[iter(kmeans_result)]*xoff)
    print("multispectral_classification() finished")
    return kmeans_result,result2D

def recreate_image(data, fileoutput, size):
    xoff, yoff = size,size
    fileout = fileoutput+".tif"

    source = gdal.Open("/Users/irene/Desktop/Teaching_Experience/dataset/cropped/x_1320_y_1930LC08_L1TP_118062_20160821_20170322_01_T1_sr_band4.tif", gdal.GA_ReadOnly)
    proj = source.GetProjection()
    geo = source.GetGeoTransform()

    # print geotransform
    driver = gdal.GetDriverByName("GTiff")

    outdata = driver.Create(fileout, xoff, yoff, 1, gdal.GDT_Float64)
    outdata.SetGeoTransform(geo)
    outdata.SetProjection(proj)

    outdata.GetRasterBand(1).WriteArray(data)
    outdata.FlushCache() ##saves to disk!!
    print ("saved on to disk for "+fileout)
    print("recreate_image() finished")

def kmeans_scratch(size, num_of_clust,  data):
    num_points = size*size
    num_clusters = num_of_clust
    cutoff = 0.2


    points = list(Point(map(float, list(rec))) for rec in data)
    clusters = kmeans(points, num_clusters, cutoff)
    return clusters

class Point(object):
    '''
    A point in n dimensional space
    '''
    def __init__(self, coords):
        '''
        coords - A list of values, one per dimension
        '''
        self.coords = coords
        self.n = len(coords)

    def __repr__(self):
        return str(self.coords)

class Cluster(object):
    '''
    A set of points and their centroid
    '''
    def __init__(self, points):
        '''
        points - A list of point objects
        '''
        if len(points) == 0:
            raise Exception("ERROR: empty cluster")

        self.points = points
        self.n = points[0].n
        for p in points:
            if p.n != self.n:
                raise Exception("ERROR: inconsistent dimensions")

        self.centroid = self.calculateCentroid()

    def __repr__(self):
        '''
        String representation of this object
        '''
        return str(self.points)

    def update(self, points):
        '''
        Returns the distance between the previous centroid and the new after
        recalculating and storing the new centroid.
        Note: Initially we expect centroids to shift around a lot and then
        gradually settle down.
        '''
        # print points
        old_centroid = self.centroid
        self.points = points
        self.centroid = self.calculateCentroid()
        # print self.points
        # print self.centroid
        shift = getDistance(old_centroid, self.centroid)
        return shift

    def calculateCentroid(self):
        '''
        Finds a virtual center point for a group of n-dimensional points
        '''
        numPoints = len(self.points)
        coords = [p.coords for p in self.points]
        unzipped = zip(*coords)
        centroid_coords = [math.fsum(dList)/numPoints for dList in unzipped]
        # print centroid_coords

        return Point(centroid_coords)

def kmeans(points, k, cutoff):
    import random

    initial = random.sample(points, k)
    clusters = [Cluster([p]) for p in initial]

    print 'initial',clusters
    loopCounter = 0

    print "kmeans()"
    while True:

        cluster_data = []
        lists = [[] for _ in clusters]
        clusterCount = len(clusters)

        loopCounter += 1

        for p in points:
            smallest_distance = getDistance(p, clusters[0].centroid)
            clusterIndex = 0
            for i in range(clusterCount - 1):
                distance = getDistance(p, clusters[i+1].centroid)
                if distance < smallest_distance:
                    smallest_distance = distance
                    clusterIndex = i+1
            lists[clusterIndex].append(p)
            cluster_data.append(clusterIndex) #label

        biggest_shift = 0.0
        for i in range(clusterCount):
            shift = clusters[i].update(lists[i])
            biggest_shift = max(biggest_shift, shift)

        print biggest_shift

        if biggest_shift < cutoff:
            print "Converged after %s iterations" % loopCounter
            break


    return cluster_data

def getDistance(a, b):
    '''
    Euclidean distance between two n-dimensional points.
    Note: This can be very slow and does not scale well
    '''

    if b.n == 0:
        print "zero"
    if a.n != b.n:
        raise Exception("ERROR: non comparable points")


    accumulatedDifference = 0.0
    for i in range(a.n):
        v1 = list(a.coords)
        v2 = list(b.coords)
        squareDifference = pow((a.coords[i]-b.coords[i]), 2)
        # squareDifference = (a.coords[i]*b.coords[i])
        accumulatedDifference += squareDifference
    distance = accumulatedDifference

    return distance

def write_label(txt,label):
    f = open(txt)
    data = [item for item in csv.reader(f)]
    f.close()


    new_data = []

    for i, item in enumerate(data):
        try:
            item.append(label[i])
        except IndexError, e:
            item.append("placeholder")
        new_data.append(item)

    f = open("result"+txt, 'w')
    csv.writer(f).writerows(new_data)
    f.close()

txtfile = "truthsample.csv"
data = csv_to_numpy(txtfile)
size = 150
clustering_result, two2D = multispectral_classification(data, size)
write_label(txtfile, clustering_result)
