import cv2
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.preprocessing import image

def load_image(image_path, grayscale=False, target_size=None):
    #载入图像，图像路径,灰度化，塑形
    pil_image = image.load_img(image_path, grayscale, target_size)
    return image.img_to_array(pil_image)#图片转化为数组

def load_detection_model(model_path):
    #检测模型,opencv自带的人脸识别模块
    detection_model = cv2.CascadeClassifier(model_path) #级联分类器
    return detection_model

def detect_faces(detection_model, gray_image_array):
    #检测人脸,探测模型,图像灰度数组，指定每个图像比例下图像大小减少的参数，指定每个候选矩形必须保留多少个邻居。
    #在输入图像中检测不同大小的对象。检测到的对象作为矩形列表返回。
    return detection_model.detectMultiScale(gray_image_array, 1.3, 5)

def draw_bounding_box(face_coordinates, image_array, color):
    #绘画,输入人脸坐标,图像数组,颜色
    x, y, w, h = face_coordinates
    #绘画矩形,图像,(左,上),(右,下),颜色
    cv2.rectangle(image_array, (x, y), (x + w, y + h), color, 2)

def apply_offsets(face_coordinates, offsets):
    x, y, width, height = face_coordinates #人脸坐标
    x_off, y_off = offsets  #扩充
    return (x - x_off, x + width + x_off, y - y_off, y + height + y_off)

def draw_text(coordinates, image_array, text, color, x_offset=0, y_offset=0,
                                                font_scale=2, thickness=2):
    #绘画文字
    x, y = coordinates[:2] #坐标第一项到第二项
    #图像，文本内容，（起点坐标），字体，字体大小，颜色，线宽，线类型
    cv2.putText(image_array, text, (x + x_offset, y + y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, color, thickness, cv2.LINE_AA)

def get_colors(num_classes):
    #获取颜色
    colors = plt.cm.hsv(np.linspace(0, 1, num_classes)).tolist()
    #返回列表,cm.hsv是图像的一种格式
    #linespace 在start和stop之间返回num_classes个均匀间隔的数据
    colors = np.asarray(colors) * 255 #转化为数组
    return colors

