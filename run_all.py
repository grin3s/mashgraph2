import numpy as np
import cv2
import matplotlib.pyplot as plt
import sys

I = cv2.imread(sys.argv[1])
plt.figure(figsize=(20,10))
plt.imshow(cv2.cvtColor(I, cv2.COLOR_BGR2RGB))

def binarize(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(img_gray, (5, 5), 0)
    #return cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    #res_image = cv2.threshold(blur, 0, 1,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
    res_image = cv2.threshold(blur, 60, 1,cv2.THRESH_BINARY)[1]
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(20,20))
    #res_image = cv2.morphologyEx(res_image, cv2.MORPH_CLOSE, kernel)
    res_image = cv2.morphologyEx(res_image, cv2.MORPH_OPEN, kernel)
    ret, labels, stats, centroids = cv2.connectedComponentsWithStats(res_image, 8, cv2.CV_32S)
    i_component = np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1
    return (labels == i_component).astype(np.uint8)

I_b = binarize(I)

import time
start = time.time()
from skeletonizer import Skeletonizer
skel_obj = Skeletonizer(I_b)
I_skel = skel_obj.find_skeleton()
print(time.time()-start)

from skeletonizer import SkelGraph

graph = SkelGraph(skel_obj, I_skel)

edges = [key for key, val in graph.degrees.items() if val == 1]
def check_leaf_length(point, l_min=100):
    path = (point, graph.connections[point][0])
    print(path)
    l = None
    if path in graph.lengths:
        l = graph.lengths[path]
    else:
        l = graph.lengths[path[::-1]]
    return l > l_min

edges = list(filter(check_leaf_length, edges))

import itertools
edge_distances = dict()
for p1, p2 in itertools.combinations(edges, 2):
    edge_distances[(p1, p2)] = np.linalg.norm((p1[0] - p2[0], p1[1] - p2[1]))

fingers = set()
for p1, p2 in sorted(edge_distances.keys(), key=edge_distances.get):
    if len(fingers) < 5:
        fingers.add(p1)
    else:
        break
    if len(fingers) < 5:
        fingers.add(p2)
    else:
        break

ordered_fingers = cv2.convexHull(np.array(list(fingers)), clockwise=True).reshape(5,2).tolist()

ordered_fingers.append(ordered_fingers[0])

i_big = np.argmax([np.linalg.norm((p1[0] - p2[0], p1[1] - p2[1])) for p1, p2 in zip(ordered_fingers[0:-1], ordered_fingers[1:])])
ordered_fingers_final = ordered_fingers[i_big + 1:-1]
ordered_fingers_final.extend(ordered_fingers[:i_big + 1])

def analyze_finger_line(point1, point2):
    I_tmp = np.zeros(I.shape, dtype=np.uint8)
    cv2.line(I_tmp, point1[::-1], point2[::-1], (1,1,1))
    n_tmp = np.sum(I_tmp[:,:,0])
    I_intersection = I_tmp[:,:,0] * I_b
    n_intersection = np.sum(I_intersection)
    if (n_intersection == n_tmp):
        return None
    else:
        I_diff = I_tmp[:,:,0] - I_intersection
        nz = np.nonzero(I_diff)
        return (nz[0][-1], nz[1][-1])

I_valleys = I.copy()
valleys = []
for p1, p2 in zip(ordered_fingers_final[0:-2], ordered_fingers_final[1:-1]):
    path1 = graph.get_path(tuple(p1), graph.connections[tuple(p1)][0])
    path2 = graph.get_path(tuple(p2), graph.connections[tuple(p2)][0])
    result = None
    i_last = 0
    while True:
        point1 = path1[i_last]
        point2 = path2[i_last]
        res = analyze_finger_line(point1, point2)
        if res is None:
            break
        else:
            i_last += 1
            result = res

    valleys.append(result)

found = False
result = None
p1 = ordered_fingers_final[-2]
p2 = ordered_fingers_final[-1]
path1 = graph.get_path(tuple(p1), graph.connections[tuple(p1)][0])
path2 = graph.get_path(tuple(p2), graph.connections[tuple(p2)][0])
for point1 in path1:
    point2 = path2[0]
    res = analyze_finger_line(point1, point2)
    if res is None:
        found = True
        break
    else:
        result = res

if not found:
    for point2 in path2:
        point1 = path1[-1]
        res = analyze_finger_line(point1, point2)
        if res is None:
            break
        else:
            result = res

valleys.append(result)
#print(valleys)
for v in valleys:
    cv2.circle(I_valleys, tuple(v[::-1]), 5, (0,0,255), -1)

I_tips = I.copy()
tips = list()
for f in ordered_fingers_final:
    path = graph.get_path(graph.connections[tuple(f)][0], tuple(f))
    path_cut = np.array(path[-29:-10]) - np.array(path[-30])
    path_cut = path_cut / np.linalg.norm(path_cut, axis=1).reshape(19, 1)
    avg_direction = np.mean(path_cut, axis=0)
    cur_f = np.array(f).astype(np.float64)
    dx = 2
    prev_f = None
    while True:
        cur_f += avg_direction
        delta = cur_f.astype(np.int)
        if I_b[delta[0], delta[1]] == 0:
            break
        else:
            prev_f = delta
    tips.append(tuple(prev_f.tolist()))

for t in tips:
    cv2.circle(I_tips, tuple(t[::-1]), 5, (0,255,0), -1)

I_final = I.copy()
for v in valleys:
    cv2.circle(I_final, tuple(v[::-1]), 5, (0,0,255), -1)

for t in tips:
    cv2.circle(I_final, tuple(t[::-1]), 5, (0,255,0), -1)

line_list = []

for i in range(0, 4):
    line_list.append(tips[i])
    line_list.append(valleys[i])


line_list.append(tips[-1])
#print(line_list)
for i in range(0, len(line_list) - 1):
    p1 = line_list[i]
    p2 = line_list[i + 1]
    cv2.line(I_final, p1[::-1], p2[::-1], (255,0,0), 1)

cv2.imwrite(sys.argv[2], I_final)

