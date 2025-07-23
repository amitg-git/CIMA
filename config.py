class Config:
    # Debug variables
    save_webcam_file = True
    save_screen_file = False
    VIEW_CAM_AT_TEST = False

    # System
    IS_PRODUCTION = True
    CAMERA_CALIB_FILE_PATH = "camera_calibration/calibration_data.joblib"

    # General application settings
    APP_TITLE = "CIMA Project"
    APP_ICON = "icon.ico"

    # File paths
    USERS_CSV_FILE = "users.csv"

    # LONGIN screen settings
    LOGIN_SCREEN_WIDTH = 400
    LOGIN_SCREEN_HEIGHT = 300

    # Menu screen settings
    MENU_SCREEN_WIDTH = 800
    MENU_SCREEN_HEIGHT = 600
    MENU_WIDTH = 200

    # CSV Analyzer screen settings
    CSV_ANALYZER_SCREEN_WIDTH = 1200
    CSV_ANALYZER_SCREEN_HEIGHT = 800
    CSV_ANALYZER_MENU_WIDTH = 200

    # Calibration screen settings
    CALIB_SCREEN_WIDTH = 800
    CALIB_SCREEN_HEIGHT = 600

    # Verification screen settings
    VERIFICATION_SCREEN_WIDTH = 800
    VERIFICATION_SCREEN_HEIGHT = 600

    @classmethod
    def get_screen_geometry(cls, screen_type):
        if screen_type == "login":
            return f"{cls.LOGIN_SCREEN_WIDTH}x{cls.LOGIN_SCREEN_HEIGHT}"
        elif screen_type == "menu":
            return f"{cls.MENU_SCREEN_WIDTH}x{cls.MENU_SCREEN_HEIGHT}"
        elif screen_type == "csv_analyzer":
            return f"{cls.CSV_ANALYZER_SCREEN_WIDTH}x{cls.CSV_ANALYZER_SCREEN_HEIGHT}"
        elif screen_type == "calibration":
            return f"{cls.CALIB_SCREEN_WIDTH}x{cls.CALIB_SCREEN_HEIGHT}"
        elif screen_type == "verification":
            return f"{cls.VERIFICATION_SCREEN_WIDTH}x{cls.VERIFICATION_SCREEN_HEIGHT}"
        else:
            raise ValueError(f"Unknown screen type: {screen_type}")


class analyze:
    EAR_OPEN_THRESHOLD = 0.16
    BLINK_DEBOUNCER_SEC = 0.2

    FACEMESH_EAR_LEFT_EYE = [
        (398, 362), (385, 384), (387, 386),
        (263, 249), (373, 374),  (380, 381)
    ]

    FACEMESH_EAR_RIGHT_EYE = [
        (33, 7), (160, 159), (158, 157),
        (173, 133), (153, 154), (144, 145)
    ]

    FOCUS_CELL_AVG_SIZE = 6
    CI_DIST_WINDOW_SIZE = 5

    CI_THRESHOLD = 20
    DI_THRESHOLD = -10

    TARGET_FPS = 0.140  # [sec]

    MAX_CI_THRESHOLD = 30  # [mm]
    MIN_CI_THRESHOLD = 7  # [mm]

    CI_THRESHOLD_CALIB_DURATION = 40  # [frames]
    DECISION_WINDOW_SIZE = 100  # [frames]
    DECISION_THRESHOLD_FOR_ALERT = 0.1  # [0 to 1]


class metric:
    FACE_METRIC_TEST_DURATION_SEC = 10  # [sec]
    CI_METRIC_SUBTEST_DURATION_FRAMES = 50  # [frames]
