#### kitti dataset orijection from lidar to top, front and rgb ####
import math

TOP_Y_MIN=-40  #40
TOP_Y_MAX=+40
TOP_X_MIN=-80
TOP_X_MAX= 80   #70.4
TOP_Z_MIN=-1.73    ###<todo> determine the correct values!
TOP_Z_MAX=0.67

TOP_X_DIVISION=0.1  #0.1
TOP_Y_DIVISION=0.1
TOP_Z_DIVISION=0.1


HORIZONTAL_FOV = math.pi
HORIZONTAL_MAX = HORIZONTAL_FOV
HORIZONTAL_MIN = 0.0
VERTICAL_FOV = math.pi * 41.33 / 180
VERTICAL_MAX = math.pi * 10.67 / 180
VERTICAL_MIN = -math.pi * 30.67 / 180
HORIZONTAL_RESOLUTION = HORIZONTAL_FOV / 512 #
VERTICAL_RESOLUTION = VERTICAL_FOV / 128 # 41.33 / 32

TR_VELO_TO_CAM = ([[ 0.                , -1.                ,  0.                ,  0.                ],
                   [ 0.                ,  0.                , -1.                ,  3.300000000000e-01],
                   [ 1.                ,  0.                ,  0.                ,  3.810000000000e-01],
                   [ 0.                ,  0.                ,  0.                ,  1.                ]])

R0_RECT        = ([[ 1.                ,  0.                ,  0.                , 0.                ],
                   [ 0.                ,  1.                ,  0.                , 0.                ],
                   [ 0.                ,  0.                ,  1.                , 0.                ],
                   [ 0.                ,  0.                ,  0.                , 1.                ]])

P3             = ([[ 1.362184692000e+03,  0.000000000000e+00,  6.205755310000e+02, 0.                ],
                   [ 0.000000000000e+00,  1.372305786000e+03,  5.618731330000e+02, 0.                ],
                   [ 0.000000000000e+00,  0.000000000000e+00,  1.000000000000e+00, 0.                ]])

P2             = ([[ 1.384621562000e+03,  0.000000000000e+00,  6.258880050000e+02, 0.                ],
                   [ 0.000000000000e+00,  1.393652271000e+03,  5.596263100000e+02, 0.                ],
                   [ 0.000000000000e+00,  0.000000000000e+00,  1.000000000000e+00, 0.                ]])

#----------------------------------
