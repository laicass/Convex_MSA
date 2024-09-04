WRITEFILE = True
LENGTH_OFFSET = 0
MU = 1.0
PERB_EPS = 0.0
ADMM_EARLY_STOP_TOGGLE = True
REINIT_W_ZERO_TOGGLE = True

NUM_THREADS = 6

# Self-defined Constants and Global Variables
MIN_DOUBLE = -1*1e99
MAX_DOUBLE = 1e99
MAX_INT = float('inf')
NUM_DNA_TYPE = 4 + 1 + 1  # A T C G + START + END
NUM_MOVEMENT = 9 + 2 + 2

START_IDX = 4
END_IDX = 5

# Algorithmic Setting
MAX_1st_FW_ITER = 100
MAX_2nd_FW_ITER = 100
MIN_ADMM_ITER = 10
MAX_ADMM_ITER = 200
FW1_GFW_EPS = 1e-6
FW2_GFW_EPS = 1e-3
# EPS_ADMM_CoZ = 1e-5
EPS_Wdiff = 0.001

# Define Scores and Other Constants
GAP_NOTATION = '-'
C_I = 1.8  # penalty of insertion
C_D = 1.8  # penalty of deletion
C_MM = 2.2 # penalty of mismatch
C_M = 0    # penalty of match
HIGH_COST = 999999
NO_COST = 0

INSERTION = 0
DELETION_A = 1
DELETION_T = 2 
DELETION_C = 3 
DELETION_G = 4
DELETION_START = 5
DELETION_END = 6
MATCH_A = 7
MATCH_T = 8 
MATCH_C = 9 
MATCH_G = 10 
MATCH_START = 11
MATCH_END = 12
UNDEFINED = 9999


INS_BASE_IDX = 0
DEL_BASE_IDX = 1  # 1-A, 2-T, 3-C, 4-G 5-* 6-#
MTH_BASE_IDX = 7  # 7-A, 8-T, 9-C, 10-G 11-* 12-#
