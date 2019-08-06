from __future__ import division
from mapping import findAngle, idFromAngle
import random

def pareto_frontier(Xs, Ys, maxX = True, maxY = False):
    '''
    This function sorts objects based on two unrelated properties
    :param Xs: list 1
    :param Ys: list 2
    :param maxX: if true, take maximum values in list 1
    :param maxY: if true, take maximum values in list 2
    :return:
    '''
    myList = sorted([[Xs[i], Ys[i],i] for i in range(len(Xs))], reverse=maxX)
    p_front = [myList[0]]
    for pair in myList[1:]:
        if maxY:
            if pair[1] >= p_front[-1][1]:
                p_front.append(pair)
        else:
            if pair[1] <= p_front[-1][1]:
                p_front.append(pair)
    p_frontX = [pair[0] for pair in p_front]
    p_frontY = [pair[1] for pair in p_front]
    p_frontIndex = [pair[2] for pair in p_front]
    return p_frontIndex

def area(roi):
    '''

    :param roi: the coordinates of bounding box
    :return: area of bounding box
    '''
    h = roi[0] - roi[2]
    w = roi[1] - roi[3]
    a = w * h
    return a


def x2angle(roi):
    '''

    :param roi: the coordinates of bounding box
    :return: converts coordinates to angle
    '''
    w = (roi[3] + roi[1]) / 2
    ref_angle = 640 / 45
    angle = (w - 320) / ref_angle
    return angle


def getAngle(local_angle, index):
    '''
    :param local_angle: angle within the image
    :param index: index of the image
    :return: total angle from the base angle
    '''
    dic = {0: -2, 1: -1, 2: 0, 3: 1, 4: 2}
    angle = local_angle + dic[index] * 45
    return angle



def output_kw(path,locationList):
    '''
    This function stores the most prominent and relevant object in each viewpoint along with its direction
    :param path: path from source to destination
    :param locationList: All the locations
    :return:
    '''
    candidate = []
    for i in range(len(path)):
        # caliulate semi_globe
        if i < len(path) - 1:
            l1 = locationList[path[i]]
            l2 = locationList[path[i + 1]]
            l1_point = [l1.lat, l2.lon]
            l2_point = [l2.lat, l2.lon]
            # print(l1_point)
            # print(l2_point)
            a = findAngle(l1_point, l2_point)
            # print(a)
            idx = idFromAngle(a)
            # print(idx)
            r1_obj, r1_roi = l1.resolveObjectsfromAngle(idx - 2, 1)
            r1_dir = [0] * len(r1_obj)
            r2_obj, r2_roi = l1.resolveObjectsfromAngle(idx - 1, 1)
            r2_dir = [1] * len(r2_obj)
            r3_obj, r3_roi = l1.resolveObjectsfromAngle(idx, 1)
            r3_dir = [2] * len(r3_obj)
            r4_obj, r4_roi = l1.resolveObjectsfromAngle(idx + 1, 1)
            r4_dir = [3] * len(r4_obj)
            r5_obj, r5_roi = l1.resolveObjectsfromAngle(idx + 2, 1)
            r5_dir = [4] * len(r5_obj)
            objects = r1_obj + r2_obj + r3_obj + r4_obj + r5_obj
            rois = r1_roi + r2_roi + r3_roi + r4_roi + r5_roi
            dirs = r1_dir + r2_dir + r3_dir + r4_dir + r5_dir
            areas = map(area, rois)
            loc_angle = map(x2angle, rois)
            loc_angle = list(loc_angle)
            angles = []
            for i in range(len(loc_angle)):
                angles.append(getAngle(loc_angle[i], dirs[i]))

            # print(objects)
            # print(rois)
            final_candidate = []
            areas = list(areas)
            pareto_idx = pareto_frontier(areas, angles)
            for i in pareto_idx:
                final_candidate.append([objects[i], dirs[i]])
            if len(final_candidate) > 2:
                final_candidate = final_candidate[1:-1]
            if len(final_candidate) > 0:
                candidate.append(random.choice(final_candidate))
    return candidate
    # measure size,proximity
    # add keywords





def get_phrase(candidate):
    '''
    This function outputs the phrases which goes as input to the NMT model
    :param candidate: a tuple of selected objects and their directions
    :return: phrase of these objects tagged with their directions
    '''
    dir_dic = {0: "left", 1: "left", 2: "straight towards", 3: "right", 4: "right"}
    phrase = ""
    for i in candidate:
        obj = i[0].split("-")[0]
        phrase = phrase + " " + dir_dic[i[1]] + " " + obj
    return phrase


a = get_phrase(candidate)
print(a)