from scipy import spatial
from sklearn import neighbors
from sklearn.preprocessing import Imputer
import numpy as np
import glob
import os, cmath
import pandas as pd

def extract_feature(folder_path):

    labeled_files = glob.glob(os.path.join(str(folder_path), "*.txt"))
    print(labeled_files)

    point_cloud = []
    point_labels = []
    label = ["sofa", "bookcase"]
    for lf in labeled_files:
        label_name = lf.split("\\")[-1].split("_")[0]
        print(label_name)
        pts = np.loadtxt(lf)[:, :3]
        point_cloud.extend(pts)
        if label_name == label[0] or label_name == label[1]:
            point_labels.extend(np.repeat(int(label.index(label_name)), len(pts)))
        else:
            point_labels.extend(np.repeat(2, len(pts)))

    point_cloud = np.array(point_cloud)
    # print(len(point_labels), len(point_cloud))
    # Features extraction
    kd = spatial.KDTree(point_cloud)
    _, vicinity = kd.query(point_cloud, k=10)
    # vicinity = kd.query_radius(point_cloud, r=0.4)

    traits = np.zeros((len(point_cloud), 6))
    for i, p in enumerate(point_cloud):
        nn = point_cloud[vicinity[i]]
        C = np.cov(nn, rowvar=False)
        s, e = np.linalg.eig(C)
        inds = np.argsort(s)[::-1]
        s, e = s[inds], e[:, inds]

        linearity = (s[0] - s[1]) / s[0]
        planarity = (s[1] - s[2]) / s[0]
        sphericity = s[2] / s[0]
        t = (s[0]*s[1]*s[2])
        omnivariance = t**(1.0/3.0)
        anisotropi = (s[0] - s[2]) / s[0]

        traits[i, 0] = linearity
        traits[i, 1] = planarity
        traits[i, 2] = sphericity
        traits[i, 3] = omnivariance
        traits[i, 4] = anisotropi
        traits[i, 5] = point_labels[i]

    np.savetxt("train_features.txt", traits)


def clasification(path_test):
    traits = np.loadtxt("train_features.txt")
    pts_test = np.loadtxt(str(path_test))[:, :3]
    # features_test = []
    kd = spatial.KDTree(pts_test)
    _, vicinity = kd.query(pts_test, k=10)

    features_test = np.zeros((len(pts_test), 6))
    for id, row in enumerate (pts_test):
        nn = pts_test[vicinity[id]]
        C = np.cov(nn, rowvar=False)
        s, e = np.linalg.eig(C)
        inds = np.argsort(s)[::-1]
        s, e = s[inds], e[:, inds]

        linearity = (s[0] - s[1]) / s[0]
        planarity = (s[1] - s[2]) / s[0]
        sphericity = s[2] / s[0]
        t = (s[0]*s[1]*s[2])
        omnivariance = t**(1.0/3.0)
        anisotropi = (s[0] - s[2]) / s[0]

        features_test[id][0]= linearity
        features_test[id][1] = planarity
        features_test[id][2] = sphericity
        features_test[id][3] = omnivariance
        features_test[id][4] = anisotropi

    np.savetxt("testfeature.txt", features_test)



def classifsklearn():
    traits = np.loadtxt("train_features.txt")
    #
    x = traits[:, :5]
    y = traits[:, 5]
    x_scaled = Imputer().fit_transform(x)

    # print(len(x), len(y))
    knn = neighbors.KNeighborsClassifier(n_neighbors=1)
    knn.fit(x_scaled, y)
    features_test = np.loadtxt("testfeature.txt")[:,:5]
    features_test_s = Imputer().fit_transform(features_test)
    label = knn.predict(features_test_s)

    np.savetxt("resultLabel.txt", label)


def main():
    folder_path= "data/Area_1/office_1/Annotations/"
    extract_feature(folder_path)
    path_test = "data/Area_1/office_1/office_1.txt"
    clasification(path_test)
    classifsklearn()

main()
