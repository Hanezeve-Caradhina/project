import numpy as np
 
 
# 双目相机参数
class stereoCamera(object):
    def __init__(self):
        # 左相机内参
        self.cam_matrix_left = np.array([[483.6312, 0, 298.5831],
                                         [0., 482.8152, 226.9264],
                                         [0., 0., 1.0000]])
        # 右相机内参
        self.cam_matrix_right = np.array([[482.7668, 0, 305.7131],
                                          [0., 482.6105, 200.1689],
                                          [0., 0., 1.0000]])
 
        # 左右相机畸变系数:[k1, k2, p1, p2, k3]
        self.distortion_l = np.array([[0.0608, 0.1244, 0.00073097, -0.00045409, -0.4115]])
        self.distortion_r = np.array([[0.0484, 0.2106, -8.8705e-04,  -0.0018, -0.5670]])
 
        # 旋转矩阵
        self.R = np.array([[0.9999, -3.8412e-04, -0.0106],
                           [3.8872e-04, 1.0000, 4.3148e-04],
                           [0.0106, -4.3557e-04, 0.9999]])
 
        # 平移矩阵
        self.T = np.array([[-60.0864], [-0.3201], [-0.7540]])
 
        # 主点列坐标的差
        self.doffs = 60.0864
 
        # 指示上述内外参是否为经过立体校正后的结果
        self.isRectified = False
 
    def setMiddleBurryParams(self):
        self.cam_matrix_left = np.array([[3997.684, 0, 225.0],
                                                            [0., 3997.684, 187.5],
                                                            [0., 0., 1.]])
        self.cam_matrix_right =  np.array([[3997.684, 0, 225.0],
                                                                [0., 3997.684, 187.5],
                                                                [0., 0., 1.]])
        self.distortion_l = np.zeros(shape=(5, 1), dtype=np.float64)
        self.distortion_r = np.zeros(shape=(5, 1), dtype=np.float64)
        self.R = np.identity(3, dtype= np.float64)
        self.T = np.array([[-193.001], [0.0], [0.0]])
        self.doffs = 131.111
        self.isRectified = True
 
 