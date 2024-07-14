import numpy as np
import cv2
import math
model_points = np.array([
    (0.0, 0.0, 0.0),
    (0.0, -330.0, -65.0),
    (-225.0, 170.0, -135.0),
    (225.0, 170.0, -135.0),
    (-150.0, -150.0, -125.0),
    (150.0, -150.0, -125.0)
])
def getHeadTiltAndCoords(size, image_points, frame_height):
    focal_length = size[1]
    center = (size[1] / 2, size[0] / 2)
    camera_matrix = np.array([[focal_length, 0, center[0]],
                              [0, focal_length, center[1]],
                              [0, 0, 1]], dtype="double")
    dist_coeffs = np.zeros((4, 1))
    _, rotation_vector, translation_vector = cv2.solvePnP(model_points, image_points,
                                                         camera_matrix, dist_coeffs,
                                                         flags=cv2.SOLVEPNP_ITERATIVE)
    nose_end_point2D, _ = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]), rotation_vector, translation_vector,
                                            camera_matrix, dist_coeffs)
    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
    euler_angles = rotationMatrixToEulerAngles(rotation_matrix)
    head_tilt_degree = euler_angles[0]
    start_point = (int(image_points[0][0]), int(image_points[0][1]))
    end_point = (int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))
    end_point_alt = (end_point[0], frame_height // 2)
    return head_tilt_degree, start_point, end_point, end_point_alt
def rotationMatrixToEulerAngles(R):
    assert (isRotationMatrix(R))
    sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-6
    if not singular:
        x = np.arctan2(R[2, 1], R[2, 2])
        y = np.arctan2(-R[2, 0], sy)
        z = np.arctan2(R[1, 0], R[0, 0])
    else:
        x = np.arctan2(-R[1, 2], R[1, 1])
        y = np.arctan2(-R[2, 0], sy)
        z = np.arctan2(R[1, 0], R[0, 0])
    x = np.degrees(x)
    y = np.degrees(y)
    z = np.degrees(z)
    x = x % 360
    if x > 180:
        x -= 360
    return np.array([x, y, z])
def isRotationMatrix(R):
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype=R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6
