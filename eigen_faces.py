"""
eigen_faces
Python code using numpy/scipy to perform face detection using eigenfaces.
Authors: Matt Brucker and Henry Rachootin
"""

import numpy as np  # Does matrix things
import scipy.io as io
import matplotlib.pyplot as plt


def parse_faces(data='faces_train'):
    faces = io.loadmat('face_detect.mat')
    imgs = faces[data]
    all_train = np.reshape(imgs, (65536, imgs.shape[2]))
    return all_train


def get_unique_labels(data='names_train'):
    labels = get_labels(data)
    unique_labels = []
    for label in labels:
        if label not in unique_labels:
            unique_labels.append(label)
    return unique_labels


def get_labels(data_type='names_train'):
    data = io.loadmat('face_detect.mat')
    all_labels = data[data_type]
    label_list = [''] * len(all_labels[0])
    for label in all_labels:
        for idx, c in enumerate(label):
            label_list[idx] += c
    label_list = [label.rstrip("\x00") for label in label_list]
    return label_list
    # print(classes)


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
    return classes


def get_closest_face(subspace_face, base_faces):
    classes = get_face_classes(base_faces)
    dists = {key: 0 for key in classes.keys()}
    for key in classes.keys():
        dist = get_face_distance(subspace_face, classes[key][:, None])
        dists[key] = dist
    return min(dists, key=dists.get)


def get_mean_face(face_set):
    """
    Returns the mean face of a given set of faces.
    """
    return np.array(np.divide(np.sum(face_set, axis=1), face_set.shape[1]))[:, None]  # Reshapes into 2D array


def get_standardized_faces(all_faces, provided_mean=None):
    """
    Returns the matrix of faces centered around the mean
    """
    if provided_mean is not None:
        mean_face = provided_mean
    else:
        mean_face = get_mean_face(all_faces)  # Gets the mean face
    imgs_stand = all_faces-mean_face  # Turns mean_face from a 1D array to a 2D array
    return imgs_stand


def find_eigen_faces(faces, num):
    """
    Finds the num most significant eigenfaces from a set of faces.
    """
    imgs_stand = get_standardized_faces(faces)  # Subtract the mean from the faces
    L = np.divide(imgs_stand, 246**(1/2))  # Build the matrix of standardized faces
    vec,val,_ = np.linalg.svd(L, full_matrices=False)
    val = np.real(val)
    u = np.real(vec)
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
    faces = parse_faces()
    print("parsed")
    eigen_faces = find_eigen_faces(faces, 200)
    print("eigen found")
    mean_face = get_mean_face(faces)
    print("mean found")
    consts = get_face_projections(get_standardized_faces(faces), eigen_faces)

    print("consts")

    test_faces_easy = parse_faces('faces_test_hard')
    print("parced")
    test_consts = get_face_projections(get_standardized_faces(test_faces_easy, mean_face), eigen_faces)
    print("test_cosnts")
    test_labels = get_labels(data_type='names_test_hard')
    print("test_labels")
    correct = 0

    for idx, face in enumerate(test_consts.T):
        guess = get_closest_face(face[:, None], consts)
        if test_labels[idx] == guess:
            correct += 1
        print(idx)
    correct /= test_faces_easy.shape[1]
    print("Correct: ", correct)
    face_reconstruct = np.dot(eigen_faces, test_consts[:,1])[:,None] + mean_face
    # test_face = get_face_projections(get_standardized_faces(faces)[:,1], eigen_faces)
    # test_result = get_closest_face(test_face, consts)
    # print(test_result)

    face = np.reshape(test_faces_easy[:,39], (256, 256))
    plt.imshow(face, cmap="gray")
    plt.show()
