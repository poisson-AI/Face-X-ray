import cv2
import dlib
import numpy as np
import sys


def annote_landmarks(im, landmarks):
    im = im.copy()
    for idx, point in enumerate(landmarks):
        pos = (point[0, 0], point[0, 1])
        cv2.putText(im, str(idx), pos,
                    fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
                    fontScale=0.4,
                    color=(0, 0, 255))
        cv2.circle(im, pos, 3, color=(0, 255, 255))
    return im


def read_im_and_landmarks(detector, predictor, im = None, fname = None, scale_fractor=1):
    if fname is not None:
        im = cv2.imread(fname, cv2.IMREAD_COLOR)
    im = cv2.resize(im, (im.shape[1] * scale_fractor,
                         im.shape[0] * scale_fractor))
    s = get_landmarks(im, detector, predictor)

    return im, s


def transformation_from_points(points1, points2):
    """
    Return an affine transformation [s * R | T] such that:

        sum || s*R*p1,i + T - p2,i||^2

    is minimized.
    """

    # Solve the procrustes problem by substracting centroids, scaling by the
    # standard deviation, and then using the SVD to calculate the rotation. See
    # the following for more details:
    # https://en.wikipedia.org/wiki/Orthogonal_Procrustes_problem

    points1 = points1.astype(np.float64)
    points2 = points2.astype(np.float64)

    c1 = np.mean(points1, axis=0)
    c2 = np.mean(points2, axis=0)
    points1 -= c1
    points2 -= c2

    s1 = np.std(points1)
    s2 = np.std(points2)
    points1 /= s1
    points2 /= s2

    U, S, Vt = np.linalg.svd(points1.T * points2)

    # The R we seek is in fact the transpose of the one given by U * Vt. This
    # is because the above formulation assumes the matrix goes on the right
    # (with row vectors) where as our solution requires the matrix to be on the
    # left (with column vectors).
    R = (U * Vt).T

    return np.vstack([np.hstack(((s2 / s1) * R,
                                       c2.T - (s2 / s1) * R * c1.T)),
                         np.matrix([0., 0., 1.])])


def draw_convex_hull(im, points, color):
    points = cv2.convexHull(points)
    cv2.fillConvexPoly(im, points, color=color)


def get_face_mask(im, landmarks, replace_points, feature_amount):
    im = np.zeros(im.shape[:2], dtype=np.float64)

    for group in replace_points:
        draw_convex_hull(im,
                         landmarks[group],
                         color=1)

    im = np.array([im, im, im]).transpose((1, 2, 0))

    im = (cv2.GaussianBlur(im, (feature_amount, feature_amount), 0) > 0) * 1.0
    im = cv2.GaussianBlur(im, (feature_amount, feature_amount), 0)

    return im


def warp_im(im, M, dshape):
    output_im = np.zeros(dshape, dtype=im.dtype)
    cv2.warpAffine(im,
                   M[:2],
                   (dshape[1], dshape[0]),
                   dst=output_im,
                   borderMode=cv2.BORDER_TRANSPARENT,
                   flags=cv2.WARP_INVERSE_MAP)
    return output_im


def correct_colors(im1, im2, landmarks1, landmarks2, colour_correct_blur_frac, blur_amount):
    # blur_amount = colour_correct_blur_frac * np.linalg.norm(
    #     np.mean(landmarks1[LEFT_EYE_POINTS], axis=0) -
    #     np.mean(landmarks2[RIGHT_EYE_POINTS], axis=0))
    blur_amount = int(blur_amount)
    if blur_amount % 2 == 0:
        blur_amount += 1

    #print(blur_amount)

    im1_blur = cv2.GaussianBlur(im1, (blur_amount, blur_amount), 0)
    im2_blur = cv2.GaussianBlur(im2, (blur_amount, blur_amount), 0)

    # cv2.imshow('im1', im1_blur)
    # cv2.imshow('im2', (128 * (im2_blur <= 1.0)).astype(im2_blur.dtype))
    # cv2.waitKey()
    # Avoid divide-by-zero errors:
    im2_blur += (128 * (im2_blur <= 1.0)).astype(im2_blur.dtype)



    return np.clip(im2.astype(np.float64) * im1_blur.astype(np.float64) /
            im2_blur.astype(np.float64), 0, 255)


class TooManyFaces(Exception):
    pass


class NoFaces(Exception):
    pass


def get_landmarks(im, detector, predictor):
    rects = detector(im, 1)

    if len(rects) > 1:
        raise TooManyFaces
    if len(rects) == 0:
        raise NoFaces
    return np.matrix([[p.x, p.y] for p in predictor(im, rects[0]).parts()])


if __name__ == '__main__':
    PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"
    SCALE_FACTOR = 1
    FEATURE_AMOUNT = 11
    FACE_POINTS = list(range(17, 68))
    MOUTH_POINTS = list(range(48, 68))
    RIGHT_BROW_POINTS = list(range(17, 22))
    LEFT_BROW_POINTS = list(range(22, 27))
    RIGHT_EYE_POINTS = list(range(36, 42))
    LEFT_EYE_POINTS = list(range(42, 48))
    NOSE_POINTS = list(range(27, 35))
    JAW_POINTS = list(range(0, 17))

    ALIGN_POINTS = (LEFT_BROW_POINTS + RIGHT_EYE_POINTS + LEFT_EYE_POINTS + RIGHT_BROW_POINTS + NOSE_POINTS + MOUTH_POINTS)

    OVERLAY_POINTS = [LEFT_EYE_POINTS + RIGHT_EYE_POINTS + LEFT_BROW_POINTS+ RIGHT_BROW_POINTS, NOSE_POINTS + MOUTH_POINTS]



    COLOUR_CORRECT_BLUR_FRAC = 0.6
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(PREDICTOR_PATH)

    im1, landmarks1 = read_im_and_landmarks(detector, predictor, fname="test_imgs/3.jpg")

    # for x in landmarks1:
    #     print(np.asarray(x)[0])
    #     cv2.circle(im1,(np.asarray(x)[0][0], np.asarray(x)[0][1]), 5, (0, 0, 255), -1)
    #
    # cv2.imshow('test', im1)
    # cv2.waitKey()
    # exit()
    im2, landmarks2 = read_im_and_landmarks(detector, predictor, fname="test_imgs/1.jpg")

    M = transformation_from_points(landmarks1[ALIGN_POINTS],
                                   landmarks2[ALIGN_POINTS])



    mask = get_face_mask(im2, landmarks2, OVERLAY_POINTS, 11)
    warped_mask = warp_im(mask, M, im1.shape)

    combined_mask = np.max([get_face_mask(im1, landmarks1, OVERLAY_POINTS, 11), warped_mask],
                              axis=0)

    warped_im2 = warp_im(im2, M, im1.shape)

    cv2.imshow('warp_img', warped_im2)
    cv2.waitKey()

    warped_corrected_im2 = correct_colors(im1, warped_im2, landmarks1, landmarks2, 0.6, 51)

    cv2.imshow('img', warped_corrected_im2.astype(np.uint8))
    output_im = im1 * (1.0 - combined_mask) + warped_corrected_im2 * combined_mask


    #print(combined_mask)
    combined_mask = np.mean(combined_mask, axis=-1, keepdims=False)
    label = 255 * 4 * (1 - combined_mask) * combined_mask

    cv2.imshow("Image1", output_im.astype(np.uint8))

    cv2.waitKey(0)

    print('ok')