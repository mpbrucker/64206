"""
eigen_faces
Python code using numpy/scipy to perform face detection using eigenfaces.
Authors: Matt Brucker and Henry Rachootin
"""

import numpy as np  # Does matrix things
import scipy.io as io
import matplotlib.pyplot as plt


def parse_faces(file_name='face_detect.mat'):
    faces = io.loadmat(file_name)
    imgs = faces['faces_train']
    all_train = np.reshape(imgs, (65536, imgs.shape[2]))
    return all_train


def get_unique_labels(file_name='face_detect.mat'):
    labels = get_labels(file_name)
    unique_labels = []
    for label in labels:
        if label not in unique_labels:
            unique_labels.append(label)
    return unique_labels


def get_labels(file_name='face_detect.mat'):
    data = io.loadmat(file_name)
    all_labels = data['names_train']
    label_list = [''] * 240
    for label in all_labels:
        for idx, c in enumerate(label):
            label_list[idx] += c
    label_list = [label.rstrip("\x00") for label in label_list]
    return label_list


def get_face_classes(components):
    labels_uniq = get_unique_labels()
    all_labels = get_labels()
    classes = dict.fromkeys(labels_uniq)
    stand_components = components / 6
    for idx, face in enumerate(stand_components.T):
        if classes[all_labels[idx]] is None:
            classes[all_labels[idx]] = face
        else:
            classes[all_labels[idx]] += face
    print(classes['Adam Selker'])
    print(get_mean_face(components[:, :6]))
    # print(classes)
    return classes


def get_closest_face(subspace_face, base_faces):
    classes = get_face_classes(base_faces)
    dists = {key: 0 for key in classes.keys()}
    for key in classes.keys():
        dists[key] = get_face_distance(subspace_face, classes[key])
    print(dists)
    return min(dists, key=dists.get)


def get_mean_face(face_set):
    """
    Returns the mean face of a given set of faces.
    """
    return np.array(np.divide(np.sum(face_set, axis=1), face_set.shape[1]))[:, None]  # Reshapes into 2D array


def get_standardized_faces(all_faces):
    """
    Returns the matrix of faces centered around the mean
    """
    mean_face = get_mean_face(all_faces)  # Gets the mean face
    imgs_stand = faces-mean_face  # Turns mean_face from a 1D array to a 2D array
    return imgs_stand


def find_eigen_faces(faces, num):
    """
    Finds the num most significant eigenfaces from a set of faces.
    """
    imgs_stand = get_standardized_faces(faces)  # Subtract the mean from the faces
    L = np.divide(imgs_stand, 246**(1/2))  # Build the matrix of standardized faces
    covar = np.dot(L.T, L)  # Find the covariance matrix of the faces
    val, vec = np.linalg.eig(covar)  # Gets the eigenvalues and vectors of the covariance matrix
    val = np.real(val)
    vec = np.real(vec)
    u = np.dot(imgs_stand, vec)
    for idx, col in enumerate(u.T):
        u[:, idx] = col/np.linalg.norm(col)
    sig_eigenfaces = u[:, 0:num]  # Take the num most significant eigenfaces
    return sig_eigenfaces


def get_face_projections(faces, eigen_faces):
    return np.dot(eigen_faces.T, faces)


def get_face_distance(face1, face2):
    diff = face1-face2
    return np.dot(diff.T, diff)


def get_face_space_distance(face, eigen_faces):
    projection = np.dot(get_face_projections(face, eigen_faces), eigen_faces)
    return np.linalg.norm(np.subtract(face, projection))


if __name__ == '__main__':
    get_labels()
    faces = parse_faces()
    eigen_faces = find_eigen_faces(faces, 200)
    mean_face = get_mean_face(faces)
    consts = get_face_projections(get_standardized_faces(faces), eigen_faces)

    print(get_closest_face(get_face_projections(faces[:, 156], eigen_faces), consts))

    face_reconstruct = np.dot(eigen_faces, consts[:,1])[:,None] + mean_face
    # test_face = get_face_projections(get_standardized_faces(faces)[:,1], eigen_faces)
    # test_result = get_closest_face(test_face, consts)
    # print(test_result)

    face = np.reshape(face_reconstruct, (256, 256))
    plt.imshow(face, cmap="gray")
    plt.show()
