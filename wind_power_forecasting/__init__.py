

# Define global variables
DATA_DIR = '../data/'
RESULTS_DIR = '../results/'
SUBMISSION_FILE = RESULTS_DIR + 'submission.csv'
RAW_DATA_DIR = DATA_DIR + 'raw/'
X_TRAIN_FILE = RAW_DATA_DIR + 'X_train_v2.csv'
Y_TRAIN_FILE = RAW_DATA_DIR + 'Y_train_sl9m6Jh.csv'
X_TEST_FILE = RAW_DATA_DIR + 'X_test_v2.csv'


# Input data labels
TIME_LABEL = 'Time'
TARGET_LABEL = 'Production'
ID_LABEL = 'ID'
WF_LABEL = 'WF'
NWP_PREFIX = 'NWP'

# Added labels
WIND_SPEED_LABEL = 'wind_speed'
WIND_VECTOR_AZIMUTH_LABEL = 'wind_vector_azimuth'
METEOROLOGICAL_WIND_DIRECTION_LABEL = 'meteorological_wind_direction'