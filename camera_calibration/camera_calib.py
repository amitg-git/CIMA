import numpy as np
import joblib
import glob
import cv2
import os
import re


def video_to_images(video_path, base_name, output_folder):
    """
    Splits an MP4 video into JPG images and saves them to the specified output folder.
    Each image is named: <base_name>_<index>.jpg
    """
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return

    frame_index = 0
    while True:
        # Read a frame from the video
        ret, frame = cap.read()

        if not ret:
            break  # No more frames to read

        # Save the frame as a JPG image
        image_path = os.path.join(output_folder, f"{base_name}_{frame_index}.jpg")
        cv2.imwrite(image_path, frame)

        frame_index += 1

    # Release the video capture object
    cap.release()
    print(f"Saved {frame_index} frames to {output_folder}")


def camera_calib(calib_images_path, *,
                 frame_to_skip: int = 1,
                 max_images: int = 100,
                 dump_file: str = "camera_calibration/calibration_data.joblib"):
    # Defining the dimensions of checkerboard
    CHECKERBOARD = (8, 6)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Defining the world coordinates for 3D points
    objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    objp[0, :, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.

    # Extracting path of individual image stored in a given directory
    images = glob.glob(f'{calib_images_path}/*.jpg')
    images.sort(key=lambda x: int(re.search(r'_(\d+)\.jpg$', x).group(1)))

    num = 0
    for i, fname in enumerate(images):
        if i % frame_to_skip != 0:
            continue
        print(f"process image: {fname}")

        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chess board corners
        # If desired number of corners are found in the image then ret = true
        ret, corners = cv2.findChessboardCorners(
            gray,
            CHECKERBOARD,
            cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE
        )

        """
        If desired number of corner are detected,
        we refine the pixel coordinates and display
        them on the images of checker board
        """
        if ret:
            objpoints.append(objp)
            # refining pixel coordinates for given 2d points.
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)

            num += 1
            if num > max_images:
                break

    if num == 0:
        print("Warning: Chessboard not found in the video!")
        return -1

    """
    Performing camera calibration by
    passing the value of known 3D points (objpoints)
    and corresponding pixel coordinates of the
    detected corners (imgpoints)
    """
    print(f"Calibrate the camera with {num} images")
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    print("Calibration Done!")

    fx = mtx[0, 0]
    fy = mtx[1, 1]

    # Get image dimensions (should match the images you calibrated with)
    width = img.shape[1]
    height = img.shape[0]

    # Normalized focal lengths (in pixels divided by image width/height)
    normalized_fx = fx / width
    normalized_fy = fy / height

    print(f"Camera matrix : \n{mtx}")
    print(f"dist : \n{dist}")
    print(f"rvecs : \n{rvecs}")
    print(f"tvecs : \n{tvecs}")
    print(f"normalized_fx: {normalized_fx}")
    print(f"normalized_fy: {normalized_fy}")

    calib_data = {
        "mtx": mtx,
        "dist": dist,
        "rvecs": rvecs,
        "tvecs": tvecs,
        "normalized_fx": normalized_fx,
        "normalized_fy": normalized_fy
    }
    joblib.dump(calib_data, dump_file)
    return 0


if __name__ == "__main__":
    calib_video = 'video/WIN_20250629_21_32_50_Pro.mp4'
    calib_images_path = 'images'

    video_to_images(calib_video, "calib", calib_images_path)
    camera_calib(calib_images_path, frame_to_skip=5)


