import math

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.ndimage import label, center_of_mass, map_coordinates
from scipy.ndimage.filters import convolve
from scipy.ndimage.filters import maximum_filter
from scipy.ndimage.morphology import generate_binary_structure
from scipy.signal import convolve2d

CORNER_SPREAD = 7
CORNER_RADIUS = 3
EIGEN_VALUE_RATIO = 0.04
PYR_SIZE = 2
TO_GREYSCALE = 'L'
GREYSCALE = 1
FLOAT = "float64"
N_COLOR = 256


def gaussian_kernel(kernel_size):
    """
    Generate a 2D Gaussian kernel.

    Parameters
    ----------
    kernel_size : int
        The size of the kernel. The kernel will be a square with sides of length
        `kernel_size`.

    Returns
    -------
    np.ndarray
        A NumPy array representing the 2D Gaussian kernel.
    """
    conv_kernel = np.array([1, 1], dtype=np.float64)[:, None]
    conv_kernel = convolve2d(conv_kernel, conv_kernel.T)
    kernel = np.array([1], dtype=np.float64)[:, None]
    for i in range(kernel_size - 1):
        kernel = convolve2d(kernel, conv_kernel, 'full')
    return kernel / kernel.sum()


def blur_spatial(img, kernel_size):
    """
    Apply a spatial blur to an image using a Gaussian kernel.

    Parameters
    ----------
    img : np.ndarray
        A 2D or 3D NumPy array representing the image. If it is a 2D array, it is
        interpreted as a grayscale image. If it is a 3D array, it is interpreted as a
        3-channel image, with the last dimension representing the channel.
    kernel_size : int
        The size of the Gaussian kernel. The kernel will be a square with sides of
        length `kernel_size`.

    Returns
    -------
    np.ndarray
        The blurred image as a NumPy array with the same shape as the input image.
    """
    kernel = gaussian_kernel(kernel_size)
    blur_img = np.zeros_like(img)
    if len(img.shape) == 2:
        blur_img = convolve2d(img, kernel, 'same', 'symm')
    else:
        for i in range(3):
            blur_img[..., i] = convolve2d(img[..., i], kernel, 'same', 'symm')
    return blur_img


def read_image(filename, representation=1):
    """
    This function reads an image file and converts it into a given representation
    @param filename: the filename of an image on disk (could be grayscale or RGB)
    @param representation: either 1 or 2 defining whether the output should be a grayscale image or an RGB image
    @return: returns an image represented by a matrix of type np.float64 normalized to the range [0,1]
    """
    img = Image.open(filename)
    img = img.convert(TO_GREYSCALE) if representation == GREYSCALE else img
    return np.asarray(img).astype(FLOAT) / (N_COLOR - 1)


def create_filter(size):
    """
    This function creates a Gaussian vector filter of size "size"
    :param size: the size of the filter
    :return: Gaussian vector filter
    """
    assert size >= 2

    base = np.array([1, 1])
    ret = base
    for i in range(size - 2):
        ret = np.convolve(ret, base.T)
    return ret.reshape((1, size)) / sum(ret)


def blur(im, filter_vec):
    """
    This function activates convolution on an image on the rows and on the columns
    :param im: a matrix
    :param filter_vec: a kernel to convolve with
    :return: the blurred image
    """
    return convolve(convolve(im, filter_vec), filter_vec.T)


def reduce(im, filter_vec):
    """
    This function reduces the size of a given image (by 2)
    :param im: the image to reduce size
    :param filter_vec: the size of the blurring vector
    :return: the smaller image resulted from the reduce
    """
    return blur(im, filter_vec)[::2].T[::2].T


def build_gaussian_pyramid(im, max_levels, filter_size):
    """
    construct a Gaussian pyramid of a given image.
    :param im: a grayscale image with double values in [0, 1]
    :param max_levels: the maximal number of levels in the resulting pyramid.
    :param filter_size: the size of the filter to be used in constructing the pyramid filter
    :return: pyr as a standard python array, filter_vec which is row vector of shape (1, filter_size)
    """
    filter_vec = gaussian_kernel(filter_size)
    pyr = [im]
    for i in range(min(max_levels - 1, int(math.log2(im.shape[0] / 16)))):
        pyr.append(reduce(pyr[-1], filter_vec))
    return pyr, filter_vec


def normalize(array):
    """
    This function computes the normalized array of the given array.
    :param array: A numpy array
    :return: an array of the same size as input, normalized
    """
    n_array = array - np.mean(array)
    norm = np.linalg.norm(n_array)
    return n_array if norm == 0 else n_array / norm


def harris_corner_detector(im):
    """
    Detects harris corners.
    Make sure the returned coordinates are x major!!!
    :param im: A 2D array representing an image.
    :return: An array with shape (N,2), where ret[i,:] are the [x,y] coordinates of the ith corner points.
    """
    derivative_filter = np.array([[1, 0, -1]])
    Ix = convolve2d(im, derivative_filter, 'same', 'symm')
    Iy = convolve2d(im, derivative_filter.T, 'same', 'symm')
    blur = lambda im: blur_spatial(im, 3)
    mats = np.dstack((blur(Ix ** 2), blur(Ix * Iy), blur(Ix * Iy), blur(Iy ** 2))).reshape((*im.shape, 2, 2))
    res = np.linalg.det(mats) - EIGEN_VALUE_RATIO * (np.trace(mats, axis1=2, axis2=3) ** 2)
    return np.argwhere(non_maximum_suppression(res).T)


def sample_descriptor(im, pos, desc_rad):
    """
    Samples descriptors at the given corners.
    :param im: A 2D array representing an image.
    :param pos: An array with shape (N,2), where pos[i,:] are the [x,y] coordinates of the ith corner point.
    :param desc_rad: "Radius" of descriptors to compute.
    :return: A 3D array with shape (N,K,K) containing the ith descriptor at desc[i,:,:].
    """
    K = (2 * desc_rad) + 1
    mops_des = []
    for x, y in pos:
        coordinates = np.meshgrid(np.linspace(y - desc_rad, y + desc_rad, K),
                                  np.linspace(x - desc_rad, x + desc_rad, K))
        samples = map_coordinates(im, coordinates, order=1, prefilter=False)
        mops_des.append(normalize(samples))
    return np.array(mops_des)


def find_features(pyr):
    """
    Detects and extracts feature points from a pyramid.
    :param pyr: Gaussian pyramid of a grayscale image having 3 levels.
    :return: A list containing:
                1) An array with shape (N,2) of [x,y] feature location per row found in the image.
                   These coordinates are provided at the pyramid level pyr[0].
                2) A feature descriptor array with shape (N,K,K)
    """

    return [spread_out_corners(pyr[0], CORNER_SPREAD, CORNER_SPREAD, CORNER_RADIUS),
            sample_descriptor(pyr[PYR_SIZE],
                              spread_out_corners(pyr[0], CORNER_SPREAD, CORNER_SPREAD, CORNER_RADIUS) / (2 ** PYR_SIZE),
                              CORNER_RADIUS)]


def match_features(desc1, desc2, min_score):
    """
    Return indices of matching descriptors.
    :param desc1: A feature descriptor array with shape (N1,K,K).
    :param desc2: A feature descriptor array with shape (N2,K,K).
    :param min_score: Minimal match score.
    :return: A list containing:
                1) An array with shape (M,) and dtype int of matching indices in desc1.
                2) An array with shape (M,) and dtype int of matching indices in desc2.
    """
    score = np.tensordot(desc1, desc2, axes=([2, 1], [2, 1]))
    score_2nd_max = np.max(np.meshgrid(np.partition(score, -2, axis=0)[-2], np.partition(score, -2, axis=1)[:, -2]),
                           axis=0)
    matches = np.argwhere(score >= np.maximum(score_2nd_max, min_score))
    return [matches[:, 0], matches[:, 1]]


def apply_homography(pos1, H12):
    """
    Apply homography to inhomogenous points.
    :param pos1: An array with shape (N,2) of [x,y] point coordinates.
    :param H12: A 3x3 homography matrix.
    :return: An array with the same shape as pos1 with [x,y] point coordinates obtained from transforming pos1 using H12.
    """
    homo_cords = np.concatenate((pos1, np.ones((1, pos1.shape[0]), dtype=int).T), axis=1) @ H12.T
    return homo_cords[:, :2] / homo_cords[:, 2:]


def ransac_homography(points1, points2, num_iter, inlier_tol, translation_only=False):
    """
    Computes homography between two sets of points using RANSAC.
    :param points1: An array with shape (N,2) containing N rows of [x,y] coordinates of matched points in image 1.
    :param points2: An array with shape (N,2) containing N rows of [x,y] coordinates of matched points in image 2.
    :param num_iter: Number of RANSAC iterations to perform.
    :param inlier_tol: inlier tolerance threshold.
    :param translation_only: see estimate rigid transform
    :return: A list containing:
                1) A 3x3 normalized homography matrix.
                2) An Array with shape (S,) where S is the number of inliers,
                    containing the indices in pos1/pos2 of the maximal set of inlier matches found.
    """
    H12s = np.apply_along_axis(lambda x: estimate_rigid_transform(points1[x], points2[x], translation_only), 1,
                               np.random.choice(len(points1), (num_iter, 2)))  # get H for each 2 random points
    inliers = np.array([np.linalg.norm(apply_homography(points1, H) - points2, axis=1) for H in H12s]) < inlier_tol
    best_iter = np.argmax(np.count_nonzero(inliers, axis=1))  # count inliers and find best random pair
    return [H12s[best_iter], np.argwhere(inliers[best_iter]).flatten()]


def display_matches(im1, im2, points1, points2, inliers):
    """
    Dispalay matching points.
    :param im1: A grayscale image.
    :param im2: A grayscale image.
    :parma pos1: An aray shape (N,2), containing N rows of [x,y] coordinates of matched points in im1.
    :param pos2: An aray shape (N,2), containing N rows of [x,y] coordinates of matched points in im2.
    :param inliers: An array with shape (S,) of inlier matches.
    """

    outliers = np.sort(np.array(list(set(range(len(points1))) - set(inliers))))
    plt.imshow(np.hstack((im1, im2)), cmap='gray', vmin=0, vmax=1)

    plt.plot(np.array([points1[outliers, 0], points2[outliers, 0] + im1.shape[1]]),
             np.array([points1[outliers, 1], points2[outliers, 1]]), mfc='r', c='b', lw=.2, ms=3, marker='o')
    plt.plot(np.array([points1[inliers, 0], points2[inliers, 0] + im1.shape[1]]),
             np.array([points1[inliers, 1], points2[inliers, 1]]), mfc='r', c='y', lw=.4, ms=7, marker='o')
    plt.show()


def accumulate_homographies(H_succesive, m):
    """
    Convert a list of succesive homographies to a
    list of homographies to a common reference frame.
    :param H_successive: A list of M-1 3x3 homography
      matrices where H_successive[i] is a homography which transforms points
      from coordinate system i to coordinate system i+1.
    :param m: Index of the coordinate system towards which we would like to
      accumulate the given homographies.
    :return: A list of M 3x3 homography matrices,
      where H2m[i] transforms points from coordinate system i to coordinate system m
    """
    ret = np.array([np.eye(3) for i in range(len(H_succesive) + 1)])
    for i in range(0, m):
        ret[:i + 1] = H_succesive[i] @ ret[:i + 1]
    for i in range(len(H_succesive), m, -1):
        ret[i:] = np.linalg.inv(H_succesive[i - 1]) @ ret[i:]
    return list(ret / ret[:, 2:, 2:])


def compute_bounding_box(homography, w, h):
    """
    computes bounding box of warped image under homography, without actually warping the image
    :param homography: homography
    :param w: width of the image
    :param h: height of the image
    :return: 2x2 array, where the first row is [x,y] of the top left corner,
     and the second row is the [x,y] of the bottom right corner
    """
    corners = np.array([[0, 0], [w - 1, 0], [0, h - 1], [w - 1, h - 1]])
    new_corners = apply_homography(corners, homography).astype(np.int)
    return np.stack((np.min(new_corners, axis=0), np.max(new_corners, axis=0)))


def warp_channel(image, homography):
    """
    Warps a 2D image with a given homography.
    :param image: a 2D image.
    :param homography: homograhpy.
    :return: A 2d warped image.
    """
    bounds = compute_bounding_box(homography, image.shape[1], image.shape[0])
    shape = bounds[1] - bounds[0]
    indices = np.meshgrid(np.linspace(bounds[0, 0], bounds[1, 0], shape[0]),
                          np.linspace(bounds[0, 1], bounds[1, 1], shape[1]))
    new_indices = apply_homography(np.dstack((indices[0], indices[1])).reshape(-1, 2), np.linalg.inv(homography))
    new_im = map_coordinates(image, np.array([new_indices[:, 1], new_indices[:, 0]]), order=1, prefilter=False)
    return new_im.reshape((shape[1], shape[0]))


def warp_image(image, homography):
    """
    Warps an RGB image with a given homography.
    :param image: an RGB image.
    :param homography: homograhpy.
    :return: A warped image.
    """
    return np.dstack([warp_channel(image[..., channel], homography) for channel in range(3)])


def filter_homographies_with_translation(homographies, minimum_right_translation):
    """
    Filters rigid transformations encoded as homographies by the amount of translation from left to right.
    :param homographies: homograhpies to filter.
    :param minimum_right_translation: amount of translation below which the transformation is discarded.
    :return: filtered homographies..
    """
    translation_over_thresh = [0]
    last = homographies[0][0, -1]
    for i in range(1, len(homographies)):
        if homographies[i][0, -1] - last > minimum_right_translation:
            translation_over_thresh.append(i)
            last = homographies[i][0, -1]
    return np.array(translation_over_thresh).astype(np.int)


def estimate_rigid_transform(points1, points2, translation_only=False):
    """
    Computes rigid transforming points1 towards points2, using least squares method.
    points1[i,:] corresponds to poins2[i,:]. In every point, the first coordinate is *x*.
    :param points1: array with shape (N,2). Holds coordinates of corresponding points from image 1.
    :param points2: array with shape (N,2). Holds coordinates of corresponding points from image 2.
    :param translation_only: whether to compute translation only. False (default) to compute rotation as well.
    :return: A 3x3 array with the computed homography.
    """
    centroid1 = points1.mean(axis=0)
    centroid2 = points2.mean(axis=0)

    if translation_only:
        rotation = np.eye(2)
        translation = centroid2 - centroid1

    else:
        centered_points1 = points1 - centroid1
        centered_points2 = points2 - centroid2

        sigma = centered_points2.T @ centered_points1
        U, _, Vt = np.linalg.svd(sigma)

        rotation = U @ Vt
        translation = -rotation @ centroid1 + centroid2

    H = np.eye(3)
    H[:2, :2] = rotation
    H[:2, 2] = translation
    return H


def non_maximum_suppression(image):
    """
    Finds local maximas of an image.
    :param image: A 2D array representing an image.
    :return: A boolean array with the same shape as the input image, where True indicates local maximum.
    """
    # Find local maximas.
    neighborhood = generate_binary_structure(2, 2)
    local_max = maximum_filter(image, footprint=neighborhood) == image
    local_max[image < (image.max() * 0.1)] = False

    # Erode areas to single points.
    lbs, num = label(local_max)
    centers = center_of_mass(local_max, lbs, np.arange(num) + 1)
    centers = np.stack(centers).round().astype(np.int)
    ret = np.zeros_like(image, dtype=np.bool)
    ret[centers[:, 0], centers[:, 1]] = True

    return ret


def spread_out_corners(im, m, n, radius):
    """
    Splits the image im to m by n rectangles and uses harris_corner_detector on each.
    :param im: A 2D array representing an image.
    :param m: Vertical number of rectangles.
    :param n: Horizontal number of rectangles.
    :param radius: Minimal distance of corner points from the boundary of the image.
    :return: An array with shape (N,2), where ret[i,:] are the [x,y] coordinates of the ith corner points.
    """
    corners = [np.empty((0, 2), dtype=np.int)]
    x_bound = np.linspace(0, im.shape[1], n + 1, dtype=np.int)
    y_bound = np.linspace(0, im.shape[0], m + 1, dtype=np.int)
    for i in range(n):
        for j in range(m):
            # Use Harris detector on every sub image.
            sub_im = im[y_bound[j]:y_bound[j + 1], x_bound[i]:x_bound[i + 1]]
            sub_corners = harris_corner_detector(sub_im)
            sub_corners += np.array([x_bound[i], y_bound[j]])[np.newaxis, :]
            corners.append(sub_corners)
    corners = np.vstack(corners)
    legit = ((corners[:, 0] > radius) & (corners[:, 0] < im.shape[1] - radius) &
             (corners[:, 1] > radius) & (corners[:, 1] < im.shape[0] - radius))
    ret = corners[legit, :]
    return ret
