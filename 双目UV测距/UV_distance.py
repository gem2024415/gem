# -*-coding: utf-8 -*-
"""
    @Author : panjq
    @E-mail : pan_jinquan@163.com
    @Date   : 2020-04-10 18:24:06
"""
import time
import os
import cv2
import argparse
import numpy as np
from PIL import Image
from shapely.geometry import Polygon
from core.utils import image_utils, file_utils
from core import camera_params, stereo_matcher
import open3d as o3d


class StereoDepth(object):
    """双目测距"""

    def __init__(self, stereo_file, width=1280, height=720, filter=True, use_open3d=True, use_pcl=False):
        """
        :param stereo_file: 双目相机内外参数配置文件
        :param width: 相机分辨率width
        :param height:相机分辨率height
        :param filter: 是否使用WLS滤波器对视差图进行滤波
        :param use_open3d: 是否使用open3d显示点云
        :param use_pcl: 是否使用PCL显示点云
        """
        self.count = 0
        self.filter = filter
        self.camera_config = camera_params.get_stereo_coefficients(stereo_file)
        self.use_pcl = use_pcl
        self.use_open3d = use_open3d
        # 初始化3D点云
        if self.use_pcl:
            # 使用open3d显示点云
            from core.utils_pcl import pcl_tools
            self.pcl_viewer = pcl_tools.PCLCloudViewer()
        if self.use_open3d:
            # 使用PCL显示点云
            from core.utils_3d import open3d_visual
            self.open3d_viewer = open3d_visual.Open3DVisual(camera_intrinsic=self.camera_config["K1"],
                                                            depth_width=width,
                                                            depth_height=height)

            self.open3d_viewer.show_image_pcd(True)
            self.open3d_viewer.show_origin_pcd(True)
            self.open3d_viewer.show_image_pcd(True)
        assert (width, height) == self.camera_config["size"], Exception("Error:{}".format(self.camera_config["size"]))

    def test_pair_image_file(self, left_file, right_file):
        """
        测试一对左右图像
        :param left_file: 左路图像文件
        :param right_file: 右路图像文件
        :return:
        """
        frameL = cv2.imread(left_file)
        framer = cv2.imread(right_file)
        point_3d = self.task(frameL, framer, waitKey=0)
        return point_3d

    def capture1(self, video):
        """
        用于采集单USB连接线的双目摄像头(左右摄像头被拼接在同一个视频中显示)
        :param video:int or str,视频路径或者摄像头ID
        :param save_dir: str,保存左右图片的路径
        """
        cap = image_utils.get_video_capture(video)
        width, height, numFrames, fps = image_utils.get_video_info(cap)
        self.count = 0
        while True:
            success, frame = cap.read()
            if not success:
                print("No more frames")
                break
            frameL = frame[:, :int(width / 2), :]
            frameR = frame[:, int(width / 2):, :]
            self.count += 1
            self.task(frameL, frameR, waitKey=5)
            if cv2.waitKey(1) & 0xFF == ord('q'):  # Get key to stop stream. Press q for exit
                break
        cap.release()
        cv2.destroyAllWindows()

    def capture2(self, left_video, right_video):
        """
        用于采集双USB连接线的双目摄像头
        :param left_video:int or str,左路视频路径或者摄像头ID
        :param right_video:int or str,右视频路径或者摄像头ID
        :return:
        """
        capL = image_utils.get_video_capture(left_video)
        capR = image_utils.get_video_capture(right_video)
        width, height, numFrames, fps = image_utils.get_video_info(capL)
        width, height, numFrames, fps = image_utils.get_video_info(capR)
        self.count = 0
        while True:
            successL, frameL = capL.read()
            successR, frameR = capR.read()
            if not (successL and successR):
                print("No more frames")
                break
            self.count += 1
            self.task(frameL, frameR, waitKey=50)
            if cv2.waitKey(1) & 0xFF == ord('q'):  # Get key to stop stream. Press q for exit
                break
        capL.release()
        capR.release()
        cv2.destroyAllWindows()

    def get_3dpoints(self, disparity, Q, scale=1.0):
        """
        计算像素点的3D坐标（左相机坐标系下）
        reprojectImageTo3D(disparity, Q),输入的Q,单位必须是毫米(mm)
        :param disparity: 视差图
        :param Q: 重投影矩阵Q=[[1, 0, 0, -cx]
                           [0, 1, 0, -cy]
                           [0, 0, 0,  f]
                           [1, 0, -1/Tx, (cx-cx`)/Tx]]
            其中f为焦距，Tx相当于平移向量T的第一个参数
        :param scale: 单位变换尺度,默认scale=1.0,单位为毫米
        :return points_3d:ndarray(np.float32),返回三维坐标points_3d，三个通道分布表示(X,Y,Z)
                    其中Z是深度图depth, 即距离,单位是毫米(mm)
        """
        # 返回三维坐标points_3d，三个通道分布表示(X,Y,Z)
        points_3d = cv2.reprojectImageTo3D(disparity, Q)
        points_3d = points_3d * scale
        points_3d = np.asarray(points_3d, dtype=np.float32)
        return points_3d

    def get_disparity(self, imgL, imgR, use_wls=True):
        """
        :param imgL: 畸变校正和立体校正后的左视图
        :param imgR：畸变校正和立体校正后的右视图
        :param use_wls：是否使用WLS滤波器对视差图进行滤波
        :return dispL:ndarray(np.float32),返回视差图
        """
        dispL = stereo_matcher.get_filter_disparity(imgL, imgR, use_wls=use_wls)
        return dispL

    def get_rectify_image(self, imgL, imgR):
        """
        畸变校正和立体校正
        根据更正map对图片进行重构
        获取用于畸变校正和立体校正的映射矩阵以及用于计算像素空间坐标的重投影矩阵
        :param imgL:
        :param imgR:
        :return:
        """
        left_map_x, left_map_y = self.camera_config["left_map_x"], self.camera_config["left_map_y"]
        right_map_x, right_map_y = self.camera_config["right_map_x"], self.camera_config["right_map_y"]
        rectifiedL = cv2.remap(imgL, left_map_x, left_map_y, cv2.INTER_LINEAR, borderValue=cv2.BORDER_CONSTANT)
        rectifiedR = cv2.remap(imgR, right_map_x, right_map_y, cv2.INTER_LINEAR, borderValue=cv2.BORDER_CONSTANT)
        return rectifiedL, rectifiedR

    def Restoring_images(self, remapped):
        src_points = np.float32([[0, 0], [1279, 0], [0, 719], [1279, 719]])
        dst_points = np.float32([[48, 48], [1167, 14], [54, 649], [1170, 669]])
        # 计算变换矩阵
        M = cv2.getPerspectiveTransform(dst_points, src_points)
        # 进行透视变换
        result = cv2.warpPerspective(remapped, M, (remapped.shape[1], remapped.shape[0]))
        cv2.imwrite('./restoredL.jpg', result)
    def Restoring_0bj(self, remapped,path):
        src_points = np.float32([[0, 0], [1279, 0], [0, 719], [1279, 719]])
        dst_points = np.float32([[48, 48], [1167, 14], [54, 649], [1170, 669]])
        # 计算变换矩阵
        M = cv2.getPerspectiveTransform(dst_points, src_points)
        # 进行透视变换
        result = cv2.warpPerspective(remapped, M, (remapped.shape[1], remapped.shape[0]))
        cv2.imwrite(path, result)
    def task(self, frameL, frameR, waitKey=5):
        """
        :param frameL: 左路视频帧图像(BGR)
        :param frameR: 右路视频帧图像(BGR)
        """
        rectifiedL, rectifiedR = self.get_rectify_image(imgL=frameL, imgR=frameR)
        cv2.imwrite('./rectifiedL.jpg', rectifiedL)
        grayL = cv2.cvtColor(rectifiedL, cv2.COLOR_BGR2GRAY)
        grayR = cv2.cvtColor(rectifiedR, cv2.COLOR_BGR2GRAY)
        dispL = self.get_disparity(grayL, grayR, self.filter)
        points_3d = self.get_3dpoints(disparity=dispL, Q=self.camera_config["Q"])
        point_3d = self.get_3dpoints(disparity=dispL, Q=self.camera_config["Q"])
        self.show_3dcloud_for_open3d(frameL, frameR, points_3d)
        self.show_2dimage(frameL, frameR, points_3d, dispL, waitKey=waitKey)
        return point_3d

    def show_3dcloud_for_open3d(self, frameL, frameR, points_3d):
        """
        使用open3d显示点云
        :param frameL:
        :param frameR:
        :param points_3d:
        :return:
        """
        if self.use_open3d:
            x, y, depth = cv2.split(points_3d)  # depth = points_3d[:, :, 2]
            self.open3d_viewer.show(color_image=frameL, depth_image=depth)

    def show_3dcloud_for_pcl(self, frameL, frameR, points_3d):
        """
        使用PCL显示点云
        :param frameL:
        :param frameR:
        :param points_3d:
        :return:
        """
        if self.use_pcl:
            self.pcl_viewer.add_3dpoints(points_3d/1000, frameL)
            self.pcl_viewer.show()

    def show_2dimage(self, frameL, frameR, points_3d, dispL, waitKey=0):
        """
        :param frameL:
        :param frameR:
        :param dispL:
        :param points_3d:
        :return:
        """
        x, y, depth = cv2.split(points_3d)  # depth = points_3d[:, :, 2]
        depth_colormap = stereo_matcher.get_visual_depth(depth)
        dispL_colormap = stereo_matcher.get_visual_disparity(dispL)
        print(dispL_colormap.shape[::-1])
        cv2.namedWindow("disparity-color", 0)
        cv2.resizeWindow("disparity-color", 1280, 720);
        image_utils.addMouseCallback("disparity-color", depth, info="depth=%fmm")
        image_utils.addMouseCallback("depth-color", depth, info="depth=%fmm")
        result = {"frameL": frameL, "frameR": frameR, "disparity": dispL_colormap, "depth": depth_colormap}

        cv2.imshow('disparity-color', dispL_colormap)

        cv2.imshow('depth-color', depth_colormap)
        key = cv2.waitKey(waitKey)
        self.save_images(result, self.count, key)

    def save_images(self, result, count, key, save_dir="./data/temp"):
        """
        :param result:
        :param count:
        :param key:
        :param save_dir:
        :return:
        """
        if key == ord('q'):
            exit(0)
        elif key == ord('c') or key == ord('s'):
            file_utils.create_dir(save_dir)
            print("save image:{:0=4d}".format(count))
            cv2.imwrite(os.path.join(save_dir, "disparity_{:0=4d}.jpg".format(count)), result["disparity"])
            cv2.imwrite(os.path.join(save_dir, "depth_{:0=4d}.jpg".format(count)), result["depth"])


def str2bool(v):
    return v.lower() in ('yes', 'true', 't', 'y', '1')


def ShapeDetection(img,imgContour,img_retified,depth):

    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    polygons = []
    rectangles = []
    rec = []
    all_rectangles = []
    all_rectangle = []
    for obj in contours:
        area = cv2.contourArea(obj)
        if area > 10:
            perimeter = cv2.arcLength(obj, True)
            approx = cv2.approxPolyDP(obj, 0.02 * perimeter, True)
            CornerNum = len(approx)
            x, y, w, h = cv2.boundingRect(approx)

            # 轮廓对象分类
            if CornerNum == 3:
                objType = "triangle"
            elif CornerNum == 4:
                if w == h:
                    objType = "Square"
                else:
                    objType = "Rectangle"
            elif CornerNum > 4:
                objType = "Circle"
            else:
                objType = "N"
            if w * h > 50:
                rectangles.append((x, y, x+w, y+h))

    cv2.imwrite('img_obj.jpg', img_retified)
    for i in range(5):
        while(True):
            x1, y1, x2, y2 = rectangles[0]
            x1_a, y1_a, x2_a, y2_a = rectangles[0]
            for rect in rectangles[1:]:
                x1_1, y1_1, x2_1, y2_1 = rect
                if (x1_1<=x2 and x1<=x1_1 and y1_1 <=y2 and y1<=y1_1 )or (
                x2_1 <=x2 and x1<=x2_1 and y2_1 <=y2 and y1<=y2_1):
                    x1 = min(x1, x1_1)
                    y1 = min(y1, y1_1)
                    x2 = max(x2, x2_1)
                    y2 = max(y2, y2_1)
                elif (x2_1 >= x1 and x1 >= x1_1 and y2_1 >= y1 and y1 >= y1_1) and (
                        x2_1 >= x2 and x2 >= x1_1 and y2_1 >= y2 and y2 >= y1_1):
                    x1 = min(x1, x1_1)
                    y1 = min(y1, y1_1)
                    x2 = max(x2, x2_1)
                    y2 = max(y2, y2_1)
                else:
                    rec.append((x1_1, y1_1, x2_1, y2_1))

            all_rectangles.append((x1, y1, x2, y2))
            rectangles.clear()
            rectangles = rec[:]
            rec.clear()
            if not rectangles:
                rectangles.clear()
                break

        rectangles = all_rectangles[:]
        all_rectangles.clear()
    all_rectangles = rectangles[:]
    while (True):
        x1, y1, x2, y2 = all_rectangles[0]
        for rect in all_rectangles[1:]:
            x1_1, y1_1, x2_1, y2_1 = rect
            if (x2_1 >= x1 and x1 >= x1_1 and y2_1 >= y1 and y1 >= y1_1) or (
                    x2_1 >= x2 and x2 >= x1_1 and y2_1 >= y2 and y2 >= y1_1):
                x1 = min(x1, x1_1)
                y1 = min(y1, y1_1)
                x2 = max(x2, x2_1)
                y2 = max(y2, y2_1)
            else:
                rec.append((x1_1, y1_1, x2_1, y2_1))
        all_rectangle.append((x1, y1, x2, y2))
        all_rectangles.clear()
        all_rectangles = rec[:]
        rec.clear()
        if not all_rectangles:
            break

    for rect in all_rectangle[0:]:
        x1_1, y1_1, x2_1, y2_1 = rect
        cv2.rectangle(imgContour, (x1_1, y1_1), (x2_1, y2_1), (0, 0, 255), 2)
        cv2.rectangle(img_retified, (x1_1, y1_1), (x2_1, y2_1), (0, 0, 255), 2)
        num = round(float(str(depth[int((y2_1 + y1_1) * 0.5), int((x2_1 + x1_1) * 0.5)] / 1000)),2)
        cv2.putText(imgContour, str(num)+"m", (x1_1+((x2_1-x1_1)//10),y1_1+((y2_1-y1_1)//10)),cv2.FONT_HERSHEY_COMPLEX,0.6,(0,0,0),1)  #绘制文字
        cv2.putText(img_retified, str(num)+"m",
                    (x1_1 + ((x2_1 - x1_1) // 10), y1_1 + ((y2_1 - y1_1) // 10)), cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 255, 0),
                    1)
def testimg(path):
    img = Image.open('path').convert('L')
    width, height = img.size
    img_fuzhi = img.resize((255, height))
    for x in range(255):
        for y in range(height):
            img_fuzhi.putpixel((x, y), 0)
    for y in range(height):
        row = list(img.getdata())[y * width:(y + 1) * width]
        for x, value in enumerate(row):
            img.putpixel((x, y), value*2+50)

    img.save('output.jpg')
def original_V_img():
    img = Image.open('./data/temp/disparity_0000.jpg').convert('L')
    width, height = img.size
    img_fuzhi = img.resize((255, height))
    for x in range(255):
        for y in range(height):
            img_fuzhi.putpixel((x, y), 0)
    for y in range(height):
        row = list(img.getdata())[y * width:(y + 1) * width]
        row_sorted = sorted(row)
        count = {}
        count_value = {}
        for value in row:
            if value in count:
                count[value] += 1
            else:
                count[value] = 1
                count_value[value] = 1
        for x, value in enumerate(row_sorted):
            if value > 0:
                if value in count_value:
                    img_fuzhi.putpixel((value, y), count[value])
            img.putpixel((x, y), count[value])

    img_fuzhi.save('original_V_output.jpg')

def V_img():
    img = Image.open('./u_img_output.jpg').convert('L')
    width, height = img.size
    img_fuzhi = img.resize((255, height))
    for x in range(255):
        for y in range(height):
            img_fuzhi.putpixel((x, y), 0)
    for y in range(height):
        row = list(img.getdata())[y * width:(y + 1) * width]
        row_sorted = sorted(row)
        count = {}
        count_value = {}
        for value in row:
            if value in count:
                count[value] += 1
            else:
                count[value] = 1
                count_value[value] = 1
        for x, value in enumerate(row_sorted):
            if value > 0:
                if value in count_value:
                    img_fuzhi.putpixel((value, y), count[value])
            img.putpixel((x, y), count[value])

    img.save('output.jpg')
    img_fuzhi.save('V_output.jpg')

def U_img():
    img = Image.open('./data/temp/disparity_0000.jpg').convert('L')
    width, height = img.size
    for x in range(width):
        col = [img.getpixel((x, y)) for y in range(height)]
        col_sorted = sorted(col)
        count = {}
        #count_value = {}
        for value in col:
            if value in count:
                count[value] += 1
            else:
                count[value] = 1
                #count_value[value] = 1
        # 将每个像素值的个数填到对应的位置
        for y, value in enumerate(col):
            if count[value] > 15:
                img.putpixel((x, y), 0)
    img_Fuzhis = Image.open('./data/temp/disparity_0000.jpg').convert('L')
    img_fuzhi = img.resize((width, 255))
    for x in range(width):
        for y in range(255):
            img_fuzhi.putpixel((x, y), 0)
    for x in range(width):
        #  部分三
        col_1 = [img_Fuzhis.getpixel((x, y)) for y in range(height)]
        col_sorted_1 = sorted(col_1)
        count12 = {}
        count_value12 = {}
        for value12 in col_1:
            if value12 in count12:
                count12[value12] += 1
            else:
                count12[value12] = 1
                count_value12[value12] = 1
        for y, value in enumerate(col_sorted_1):
            if value in count_value12:
                img_fuzhi.putpixel((x, value), count12[value])
    img.save('u_img_output.jpg')
    img_fuzhi.save('original_U_output.jpg')
def U_image():

    img_Fuzhis = Image.open('./Done.jpg').convert('L')
    width, height = img_Fuzhis.size
    img_fuzhi = img_Fuzhis.resize((width, 255))
    for x in range(width):
        for y in range(255):
            img_fuzhi.putpixel((x, y), 0)
    for x in range(width):
        #  部分三
        col_1 = [img_Fuzhis.getpixel((x, y)) for y in range(height)]
        col_sorted_1 = sorted(col_1)
        count12 = {}
        count_value12 = {}
        for value12 in col_1:
            if value12 in count12:
                count12[value12] += 1
            else:
                count12[value12] = 1
                count_value12[value12] = 1
        for y, value in enumerate(col_sorted_1):
            if value in count_value12:
                if value > 3:
                    if count12[value]>20:
                        img_fuzhi.putpixel((x, value), count12[value])

    img_fuzhi.save('U_output.jpg')

def get_parser():
    stereo_file = r"configs\lenacv-camera\stereo_cam.yml"
    left_video = r"data\lenacv-video\left_video.avi"
    right_video = r"data\lenacv-video\right_video.avi"
    left_file = r"data\lenacv-camera\left_1.jpeg"
    right_file = r"data\lenacv-camera\right_1.jpeg"
    parser = argparse.ArgumentParser(description='Camera calibration')
    parser.add_argument('--stereo_file', type=str, default=stereo_file, help='stereo calibration file')
    parser.add_argument('--left_video', default=left_video, help='left video file or camera ID')
    parser.add_argument('--right_video', default=right_video, help='right video file or camera ID')
    parser.add_argument('--left_file', type=str, default=left_file, help='left image file')
    parser.add_argument('--right_file', type=str, default=right_file, help='right image file')
    parser.add_argument('--filter', type=str2bool, nargs='?', default=True, help='use disparity filter')
    return parser
def fit_line_ransac(points, iterations=100, threshold=5):
    """
    使用 RANSAC 算法拟合直线
    :param points: 点集，每个点为 (x, y) 的二元组
    :param iterations: 迭代次数
    :param threshold: 阈值，用于判断点是否属于拟合的直线
    :return: 直线斜率和截距
    """
    num_points = len(points)
    best_m, best_b = None, None
    best_inliers = []

    for i in range(iterations):
        sample = np.random.choice(num_points, 2, replace=False)
        p1, p2 = points[sample[0]], points[sample[1]]
        if (p2[0] - p1[0]) != 0:

            m = (p2[1] - p1[1]) / (p2[0] - p1[0])
            b = p1[1] - m * p1[0]
            distances = []
            for j in range(num_points):
                point = points[j]
                distance = abs(point[1] - m * point[0] - b) / np.sqrt(m ** 2 + 1)
                distances.append(distance)
            inliers = np.where(np.array(distances) < threshold)[0]
            if len(inliers) > len(best_inliers):
                best_inliers = inliers
                best_m, best_b = m, b

    return best_m, best_b

def RANSAC():
    disparity_map = cv2.imread('V_output.jpg', cv2.IMREAD_GRAYSCALE)
    road_mask = (disparity_map > 20) & (disparity_map < 254)
    points = []
    for y in range(disparity_map.shape[0]):
        for x in range(disparity_map.shape[1]):
            if x > 5:
                if road_mask[y, x]:
                    points.append((x, y))

    m, b = fit_line_ransac(points)
    img = Image.open('./data/temp/disparity_0000.jpg').convert('L')
    imgL = Image.open('./rectifiedL.jpg')
    width, height = img.size
    color_to_make_transparent = (0, 0, 255)
    m1 = (m+0.27735)/(1-m*0.27735)
    b1 = 720-m1*(720-b)/m
    for y in range(height):
        row = list(img.getdata())[y * width:(y + 1) * width]
        row_sorted = sorted(row)
        count = {}
        count_value = {}
        for value in row:
            if value in count:
                count[value] += 1
            else:
                count[value] = 1
                count_value[value] = 1
        x = (y-b)/m
        x_int = int(x)
        if x_int > 0:
            for x, value in enumerate(row):
                if value < x_int:
                    imgL.putpixel((x, y), (0, 0, 0, 0))
                elif value == x_int:
                    imgL.putpixel((x, y), (0, 0, 0, 0))
        for x, value in enumerate(row):
            if value < x_int+3 or value >= (720-b1)/m1:
               img.putpixel((x, y), 0)
            if value < 4:
                img.putpixel((x, y), 0)
    img.save('Done.jpg')
    imgL.save('DoneL.jpg')
    line_pts = np.array([(0, int(b)), (disparity_map.shape[1], int(m * disparity_map.shape[1] + b))])
    cv2.line(disparity_map, tuple(line_pts[0]), tuple(line_pts[1]), (255, 255, 255), 2)
    cv2.imwrite('./Disparity_Map.jpg', disparity_map)


def find_obstacle_lines(udisp, Xgap):
    rows, cols = udisp.shape
    lines = []  # 存储障碍物线段集合
    line = None  # 当前障碍物线段
    gap = 0  # 当前间隔
    for row in range(rows):
        for col in range(cols):
            intensity = udisp[row][col]
            if intensity > 0:
                if line is None:
                    line = (row, col, col)
                else:
                    line = (row, line[1], col)
                gap = 0
            else:
                if line is not None:
                    gap += 1
                    if gap > Xgap:
                        lines.append(line)
                        line = None
                        gap = 0
                    else:
                        line = (row, line[1], line[2])
        if line is not None:
            lines.append(line)
            line = None
            gap = 0

    # 将障碍物线段画出来
    img = cv2.cvtColor(udisp, cv2.COLOR_GRAY2RGB)
    for line in lines:
        cv2.line(img, (line[1], line[0]), (line[2], line[0]), (0, 0, 255), 1)

    return lines, img

def overlap(line1, line2):
    _, left1, right1 = line1
    _, left2, right2 = line2
    overlap = max(0, min(right1, right2) - max(left1, left2))
    a = max(right1 - left1, right2 - left2)
    if a != 0:
        return overlap / a
    else:
        return 0

def merge_obstacle_lines(lines, threshold):
    n = len(lines)
    labels = [i for i in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            if overlap(lines[i], lines[j]) >= threshold:
                labels[j] = labels[i]
    groups = {}
    for i in range(n):
        if labels[i] not in groups:
            groups[labels[i]] = []
        groups[labels[i]].append(lines[i])
    merged_lines = []
    for group in groups.values():
        row = group[0][0]
        left = min(line[1] for line in group)
        right = max(line[2] for line in group)
        merged_lines.append((row, left, right))

    return merged_lines

def rectangle(obstacles):
    rectangles = []
    dis_vec = []
    temp = 0
    i = 0
    flag_max = 0
    for obs in obstacles:
        dis = obs[2]-obs[1]
        dis_vec.append(dis)
        if temp < dis:
            temp = dis
            max_dis = dis
            flag_max = i
        i = i + 1
    obs1 = obstacles[flag_max]
    flag = 1

    while(flag):
        max_row = obstacles[flag_max][0]
        min_row = obstacles[flag_max][0]
        num_add = []
        num_sub = []
        surplus = []
        max_col = obstacles[flag_max][2]
        min_col = obstacles[flag_max][1]
        for obs in obstacles:
            if obs[1] >= obstacles[flag_max][2] or obs[2] <= obstacles[flag_max][1]:
                surplus.append(obs)
            else:
                if obs[2] > obstacles[flag_max][2]:
                    max_col = obs[2]
                if obs[1] < obstacles[flag_max][1]:
                    min_col = obs[1]
                if max_row < obs[0]:
                    if obs[0]-obstacles[flag_max][0] < 6:
                        num_add.append(obs[0] - obstacles[flag_max][0])
                    else:
                        surplus.append(obs)
                elif min_row > obs[0]:
                    if obstacles[flag_max][0]-obs[0] < 6:
                        num_sub.append(obs[0] - obstacles[flag_max][0])
                    else:
                        surplus.append(obs)
        for i_max in range(6):
            if i_max+1 in num_add:
                max_row = i_max+obstacles[flag_max][0]
            else:
                break
        for i_min in range(6):
            if (-(i_min+1)) in num_sub:
                min_row = -i_min+obstacles[flag_max][0]
            else:
                break
        rectangles.append(((min_col, min_row), (max_col, max_row)))
        obstacles.clear()
        obstacles = surplus[:]
        if not surplus:
            surplus.clear()
            break
        surplus.clear()
        temp = 0
        i = 0
        flag_max = 0
        dis_vec.clear()
        for obs in obstacles:
            dis = obs[2] - obs[1]
            dis_vec.append(dis)
            if temp < dis:
                temp = dis
                max_dis = dis
                flag_max = i
            i = i + 1
        if temp == 0:
            break
    return rectangles



def Draw_rectangle(dep_finnal_vec,temprect,rectangles, path, path2, depth):
    img = Image.open(path).convert('L')
    img_U = Image.open('./U_output.jpg')
    img_obj = cv2.imread(path2)
    img_rect = cv2.imread(path)
    for rect in rectangles[0:]:
        point_1, point_2 = rect
        width, height = img.size
        y_max = point_2[1]
        y_min = point_1[1]
        y_vec = []
        if y_max > y_min:
            max_num = 0
            max_value_y = 0
            for x in range(point_1[0], point_2[0]+1):
                col_2 = [img_U.getpixel((x, y)) for y in range(255)]
                for y, value in enumerate(col_2):
                    if value > 3:
                        if max_num < value:
                            # 取最大个数的深度值
                            max_num = value
                            max_value_y = y
            x_depth = 0
            y_depth = 0
            image_value = 0
            for x in range(point_1[0], point_2[0]+1):
                col_1 = [img.getpixel((x, y)) for y in range(height)]
                for y, value in enumerate(col_1):
                    if value > 3:
                        if value >= point_1[1] and value <= point_2[1]:
                            y_vec.append(y)
                        if value == max_value_y:
                            x_depth = x
                            y_depth = y
                            image_value = value
            dep = depth[y_depth, x_depth]/(255-image_value)
            dep_finnal = dep*(255-(point_2[1]+point_1[1])/2)
            dep_finnal_vec.append(dep_finnal)
            flag_y = 0
            for y in range(height):
                if y in y_vec:
                    if flag_y == 0:
                        y_min = y
                        flag_y = 1
                    else:
                        y_max = y
                else:
                    if flag_y == 0:
                        y = y
                    else:
                        break

            y_vec.clear()
            temprect.append(((point_1[0], y_min), (point_2[0], y_max)))
            cv2.rectangle(img_rect, (point_1[0], y_min), (point_2[0], y_max), (0, 0, 255), 2)
            cv2.rectangle(img_obj, (point_1[0], y_min), (point_2[0], y_max), (0, 0, 255), 2)
            x1_1, x2_1 = point_1[0], point_2[0]
            y1_1, y2_1 = y_min, y_max
            num = round(float(str(dep_finnal / 1000)), 2)
            cv2.putText(img_rect, str(num) + "m", (x1_1 + ((x2_1 - x1_1) // 10), y1_1 + ((y2_1 - y1_1) // 10)),
                        cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 255, 0), 1)  # 绘制文字
            cv2.putText(img_obj, str(num) + "m", (x1_1 + ((x2_1 - x1_1) // 10), y1_1 + ((y2_1 - y1_1) // 10)),
                        cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 255, 0), 1)  # 绘制文字

    cv2.imshow('img_rect', img_obj)
    cv2.imwrite('./img_rect.jpg', img_obj)


if __name__ == '__main__':
    testimg()
    args = get_parser().parse_args()
    stereo = StereoDepth(args.stereo_file, filter=args.filter)
    if args.left_video is not None and args.right_video is not None:
        stereo.capture2(left_video=args.left_video, right_video=args.right_video)
    elif args.left_video is not None:
        stereo.capture1(video=args.left_video)
    elif args.right_video is not None:
        stereo.capture1(video=args.right_video)
    if args.left_file and args.right_file:
        start = time.perf_counter()
        point_3d = stereo.test_pair_image_file(args.left_file, args.right_file)
        x, y, depth = cv2.split(point_3d)
        U_img()
        V_img()
        original_V_img()
        RANSAC()
        U_image()
        udisp = cv2.imread('U_output.jpg')
        udispGray = cv2.cvtColor(udisp, cv2.COLOR_RGB2GRAY)
        obstacle_segments, img_line = find_obstacle_lines(udispGray, 4)
        cv2.imwrite('img_line.jpg', img_line)
        rectangles = rectangle(obstacle_segments)
        for rect in rectangles[0:]:
            point_1, point_2 = rect
            cv2.rectangle(udisp, point_1, point_2, (0, 0, 255), 2)
        cv2.imshow('merged_lines', udisp)
        cv2.imwrite('img_line_rectangle.jpg', udisp)
        temprect = []
        dep_finnal_vec = []
        Draw_rectangle(dep_finnal_vec, temprect, rectangles, './Done.jpg','./rectifiedL.jpg', depth)
        img_restoring = cv2.imread('./DoneL.jpg')
        stereo.Restoring_images(img_restoring)
        img_retified = cv2.imread('./img_rect.jpg')
        stereo.Restoring_0bj(img_retified, './img_finally_obj.jpg')
        end = time.perf_counter()
        print('运行时间为：{}秒'.format(end - start))
