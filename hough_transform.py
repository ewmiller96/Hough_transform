from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from skimage.transform import hough_line, hough_line_peaks
import random
import math as m
import scipy as sp

file_location = "mu10GeV_4particles" # change to the location of the truth and clusters files
event_number = 5 # change to the number that follows "clusters" and "truth"

# these functions load events and return their hit info in pandas data frames, labels_2 and labels_3 re for rare cases of hits shared by tracks "mu10GeV/clusters_"+str(event)+".csv"

def load_event(preface, event_num):
    hits = pd.read_csv(preface+ "/clusters_"+str(event_num)+".csv", names = ["x", "y", "z", "labels", "labels_2", "labels_3", "track"])
    truth = pd.read_csv(preface+ "/truth_"+str(event_num)+".csv", names = ["labels", "pt", "eta", "phi", "e"])
    hits.fillna(0.0, inplace=True)
    truth.fillna(0.0, inplace=True)
    return hits, truth

def conformal_map(hits, x0=0, y0=0):
    '''maps hits from x, y, z coords to u, v, z such that circles passing close
    to x=x0, y=y0 are mapped to straight lines in u,v'''

    x = hits.x.values
    y = hits.y.values
    u_list = []
    v_list = []
    for i in range (0, len(hits.index.values)):
        r_sq = ((x[i]-x0)**2 + (y[i]-y0)**2)
        u = (x[i]-x0)/r_sq
        v = (y[i]-y0)/r_sq
        u_list.append(u)
        v_list.append(v)

    hits["u"] = pd.Series(u_list, index = hits.index)
    hits["v"] = pd.Series(v_list, index = hits.index)

def bin_plot(hits, num_bins=200, binary=False, mapped=True):
    '''makes num_bins X num_bins histogram of u,v (or x,y if mapped == False)
    plane where the value of each cell is the nuber of hits that fall inside it,
    or if binary == True the cell values are 1 if any hit falls in it or 0
    otherwise'''

    if mapped:
        hist, x_edges, y_edges = np.histogram2d(getattr(hits, "v"), getattr(hits, "u"), bins=[num_bins, num_bins], range=[[-0.03,0.03],[-0.03,0.03]])
    else:
        hist, x_edges, y_edges = np.histogram2d(getattr(hits, "y"), getattr(hits, "x"), bins=[num_bins, num_bins], range=[[-1100,1100],[-1100,1100]])

    if binary:
        mask = hist != 0
        hist[mask] = 1
    return hist

def get_lines(hist, num_bins, num_angles=1000, thresh=None, max_dist=-1):
    '''finds lines in the binned hist using skimage hough transform. if
    max_dist > 0 then only lines whose closest approach to the origin within
    this distance will be returned. num_bins should be the same as is used in
    bin_plot()'''

    scaling = 0.06/num_bins #used to transform from bin coordinates to detector coordinates
    theta = np.linspace(-np.pi/2, np.pi/2, num=num_angles) # list of angles that the HT will check

    # performs HT
    h, theta, d = hough_line(hist, theta=theta)
    h_peak, theta_peak, d_peak = hough_line_peaks(h, theta, d, threshold=thresh)

    gradients = np.vectorize(lambda a: -np.tan((np.pi/2 - a)))(theta_peak)

    scaled_intercept_list = [] #gets y intercept of lines in global detector coordinates
    for angle, dist in zip(theta_peak, d_peak):
        intercept = (dist-(num_bins/2)*np.cos(angle)) / np.sin(angle)
        scaled_intercept = scaling*intercept -0.03
        scaled_intercept_list.append(scaled_intercept)

    scaled_dist = [] # gets distance of closest approach to origin of the lines in global detector coords
    for angle, c, m in zip(theta_peak, scaled_intercept_list, gradients):
        d = c/(np.sin(angle) - m*np.cos(angle))
        scaled_dist.append(d)

    lines = pd.DataFrame(np.column_stack([theta_peak, d_peak, gradients, scaled_intercept_list, scaled_dist]), columns = ["angle", "histogram_distance", "gradient", "intercept", "origin_distance"])

    if max_dist>=0.0:
        within_dist = abs(lines.origin_distance.values) <= max_dist
        lines = lines[within_dist]
        lines.reset_index(drop=True, inplace=True)

    return lines

def get_line_tracks(hits, lines, max_dist=np.inf):
    '''assigns hits to line (track) that they are closest to. if no line is
    within maximum distance (max_dist), the hit is considered "noise" and given
    track label of -1'''

    max_dist_sq = max_dist**2

    for i in range(0, len(hits.index.values)): #loop over all hits
        u = hits.u.values[i]
        v = hits.v.values[i]

        dist_square_list = []
        for j in range(0, len(lines.index.values)): #this loop gets distance of point to each line and stores them in a list
            m = lines.gradient.values[j]
            c = lines.intercept.values[j]

            dist_sq = (v - (c+m*u))**2 / (1.0+m**2)
            dist_square_list.append(dist_sq)

        minimum = min(dist_square_list) #finds the line that the hit is closest to

        if minimum <= max_dist_sq: #check tht the hit to closest line distance is within the distance threshold and sets track label to corresponding line number
            track = np.where(dist_square_list == minimum)[0]
        else:
            track = -1.0

        hits.track[i] = track


## code below runs each step and make plots of the lines and the found tracks ##

data, truth = load_event(file_location, event_number)

conformal_map(data)
hist = bin_plot(data, num_bins=300, binary=True, mapped=True)
lines = get_lines(hist, num_bins = 300, num_angles=500, max_dist=1)
get_line_tracks(data, lines, max_dist=1)

fig, ax = plt.subplots() #hits coloured according to their assigned track
for m, c in zip(lines.gradient.values, lines.intercept.values):
        y0 = m*(-0.03) + c
        y1 = m*(0.03) + c
        ax.plot((-0.03, 0.03), (y0, y1), '-r')
ax.set_xlim((-0.03, 0.03))
ax.set_ylim((-0.03, 0.03))
ax.set_title('Detected lines')
ax.set_xlabel("u")
ax.set_ylabel("v")

plt.scatter(data.u, data.v, c=data.track)
plt.savefig("lines_with_hits")

plt.figure(2) #hits coloured according to their true true track
plt.scatter(data.u, data.v, c=data.labels % 1000)
plt.xlabel("u")
plt.ylabel("v")

fig, ax = plt.subplots() #plot of the histogram with found lines overlayed
ax.imshow(hist)
for angle, dist in zip(lines.angle.values, lines.histogram_distance.values):
        y0 = (dist - 0 * np.cos(angle)) / np.sin(angle)
        y1 = (dist - hist.shape[1] * np.cos(angle)) / np.sin(angle)
        ax.plot((0, hist.shape[1]), (y0, y1), '-r')
ax.set_xlim((0, hist.shape[1]))
ax.set_ylim((0, hist.shape[0]))
ax.set_title('Detected lines')
ax.set_xlabel("u")
ax.set_ylabel("v")

plt.savefig("lines_with_histogram")
plt.show()
