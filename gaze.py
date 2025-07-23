import cv2
import numpy as np
from helpers import relative, relativeT


def calc_ear(frame, mp_results, eye_connections):
    if not mp_results.multi_face_landmarks:
        return 0.0

    face_landmarks = mp_results.multi_face_landmarks[0]

    h, w, _ = frame.shape
    xy = []
    for pair in eye_connections:
        p = pair[0]
        x = int(face_landmarks.landmark[p].x * w)
        y = int(face_landmarks.landmark[p].y * h)
        xy.append(np.array([x, y]))

    p15 = np.linalg.norm(xy[1] - xy[5])
    p24 = np.linalg.norm(xy[2] - xy[4])
    p03 = np.linalg.norm(xy[0] - xy[3])

    ear = (p15 + p24) / (2 * p03)
    return ear


def calc_iris_depth(frame, mp_results, eye_connections, *, normalized_focal_x):
    """
    Base on URL:
    https://medium.com/@susanne.thierfelder/create-your-own-depth-measuring-tool-with-mediapipe-facemesh-in-javascript-ae90abae2362
    """
    if not mp_results.multi_face_landmarks:
        return float('inf')

    face_landmarks = mp_results.multi_face_landmarks[0]
    height, width, _ = frame.shape

    iris_min_x = float('inf')
    iris_max_x = float('-inf')

    for point in eye_connections:
        point0 = face_landmarks.landmark[point[0]]
        iris_min_x = min(iris_min_x, point0.x * width)
        iris_max_x = max(iris_max_x, point0.x * width)

    dx = iris_max_x - iris_min_x   # unit [pixels]
    dX = 11.7  # unit [mm]

    fx = width * normalized_focal_x  # unit[pixels]
    dZ = (fx * (dX / dx)) / 10.0  # unit [cm]
    return dZ  # unit [cm]


def gaze(frame, mp_results):
    """
    The gaze function gets an image and face landmarks from mediapipe framework.
    The function draws the gaze direction into the frame.
    Source:
        https://medium.com/@amit.aflalo2/eye-gaze-estimation-using-a-webcam-in-100-lines-of-code-570d4683fe23
    """
    if not mp_results.multi_face_landmarks:
        return None

    points = mp_results.multi_face_landmarks[0]
    '''
    2D image points (unit [pixels]).
    relative takes mediapipe points that is normalized to [-1, 1] and returns image points
    at (x,y) format
    '''
    image_points = np.array([
        relative(points.landmark[4], frame.shape),  # Nose tip
        relative(points.landmark[152], frame.shape),  # Chin
        relative(points.landmark[263], frame.shape),  # Left eye - left corner
        relative(points.landmark[33], frame.shape),  # Right eye - right corner
        relative(points.landmark[287], frame.shape),  # Left Mouth corner
        relative(points.landmark[57], frame.shape)  # Right mouth corner
    ], dtype="double")

    '''
    2D image points (unit [pixels]).
    relativeT takes mediapipe points that is normalized to [-1, 1] and returns image points
    at (x,y,0) format
    '''
    image_points1 = np.array([
        relativeT(points.landmark[4], frame.shape),  # Nose tip
        relativeT(points.landmark[152], frame.shape),  # Chin
        relativeT(points.landmark[263], frame.shape),  # Left eye, left corner
        relativeT(points.landmark[33], frame.shape),  # Right eye, right corner
        relativeT(points.landmark[287], frame.shape),  # Left Mouth corner
        relativeT(points.landmark[57], frame.shape)  # Right mouth corner
    ], dtype="double")

    # 3D model points (unit [mm]).
    model_points = np.array([
        (0.0, 0.0, 0.0),  # Nose tip
        (0, -63.6, -12.5),  # Chin
        (-43.3, 32.7, -26),  # Left eye, left corner
        (43.3, 32.7, -26),  # Right eye, right corner
        (-28.9, -28.9, -24.1),  # Left Mouth corner
        (28.9, -28.9, -24.1)  # Right mouth corner
    ])

    '''
    3D model eye points (unit [mm]).
    The center of the eye ball
    '''
    Eye_ball_center_right = np.array([[-29.05], [32.7], [-39.5]])
    Eye_ball_center_left = np.array([[29.05], [32.7], [-39.5]])

    '''
    camera matrix estimation
    Source (camera intrinsic matrix):
        https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html
    '''
    fx = frame.shape[1]  # unit [pixels]
    center = (frame.shape[1] / 2, frame.shape[0] / 2)
    camera_matrix = np.array(
        [[fx, 0, center[0]],
         [0, fx, center[1]],
         [0, 0, 1]], dtype="double"
    )

    '''
    - translation_vector from [mm] coord-system (model_points) to [pixels] coord-system (image_points)
    '''
    dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion
    (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix,
                                                                  dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)

    # 2d pupil location in pixels  (unit [pixels]).
    left_pupil = relative(points.landmark[468], frame.shape)
    right_pupil = relative(points.landmark[473], frame.shape)

    # Transformation between image point to world point (unit: from [pixels] to [mm])
    _, transformation, _ = cv2.estimateAffine3D(image_points1, model_points)  # image to world transformation

    trans_cache = (camera_matrix, dist_coeffs, rotation_vector, translation_vector, transformation)
    if transformation is not None:  # if estimateAffine3D succeeded
        # project pupil image point into 3d world point
        pupil_world_cord = transformation @ np.array([[left_pupil[0], left_pupil[1], 0, 1]]).T

        # 3D gaze point (10 is arbitrary value denoting gaze distance)
        S = Eye_ball_center_left + (pupil_world_cord - Eye_ball_center_left) * 10

        # Project a 3D gaze direction onto the image plane.
        (eye_pupil2D, _) = cv2.projectPoints((int(S[0]), int(S[1]), int(S[2])), rotation_vector,
                                             translation_vector, camera_matrix, dist_coeffs)
        # project 3D head pose into the image plane
        (head_pose, _) = cv2.projectPoints((int(pupil_world_cord[0]), int(pupil_world_cord[1]), int(40)),
                                           rotation_vector,
                                           translation_vector, camera_matrix, dist_coeffs)

        # left_gaze = _calc_gaze_direction(left_pupil, Eye_ball_center_left, trans_cache)
        # right_gaze = _calc_gaze_direction(right_pupil, Eye_ball_center_right, trans_cache)
        # delta_gaze = abs(left_gaze - right_gaze)

        return head_pose[0][0]
    return None


def _calc_gaze_direction(pupil_loc, eye_ball_center, trans_cache):
    # unpack translation cache
    (camera_matrix, dist_coeffs, rotation_vector, translation_vector, transformation) = trans_cache

    # project pupil image point into 3d world point
    pupil_world_cord = transformation @ np.array([[pupil_loc[0], pupil_loc[1], 0, 1]]).T
    # pupil_world_cord[2] = 40  # estimated depth (should be negative)

    # 3D gaze point (10 is arbitrary value denoting gaze distance)
    s = eye_ball_center + (pupil_world_cord - eye_ball_center) * 10

    # Project a 3D gaze direction onto the image plane.
    (eye_pupil2D, _) = cv2.projectPoints(tuple(map(int, s)),
                                         rotation_vector, translation_vector, camera_matrix, dist_coeffs)
    # project 3D head pose into the image plane
    (head_pose, _) = cv2.projectPoints(tuple(map(int, pupil_world_cord)),
                                       rotation_vector, translation_vector, camera_matrix, dist_coeffs)
    # correct gaze for head rotation
    gaze_loc = pupil_loc + (eye_pupil2D[0][0] - pupil_loc) - (head_pose[0][0] - pupil_loc)
    return gaze_loc
