from PyQt5.QtGui import QFont, QPixmap, QBrush, QColor, QImage
from PyQt5.QtWidgets import QTabWidget, QWidget, QLabel, QVBoxLayout, QPushButton, QHBoxLayout, QTableWidget
from PyQt5.QtWidgets import QHeaderView, QFileDialog, QTableWidgetItem, QApplication
import sys

from tensorflow.keras.models import load_model
from utils.data_manage import *
from utils.inference import *

emotion_labels = get_labels('fer2013') #获取情感标签
emotion_offsets = (0, 0)

detection_model_path = './model/haarcascade_frontalface_default.xml' #检测文件
emotion_model_path = 'model/fer2013_model1.hdf5'  #权重文件

# 载入模型
face_detection = load_detection_model(detection_model_path)       #返回检测模型，初始化
emotion_classifier = load_model(emotion_model_path, compile=False) #加载权值模型文件
emotion_target_size = emotion_classifier.input_shape[1:3]  #获得输入模型形状


class VideoBox(QTabWidget): #封装类
    def __init__(self): #初始化
        QWidget.__init__(self)
        self.frame = None
        self.line_point = {}
        self.frame_played = 0
        self.emotion_map_init = {'angry': 0, 'disgust': 0, 'fear': 0, 'happy': 0, 'sad': 0,
                                 'surprise': 0, 'neutral': 0}
        self.emotion_map = ''
        self.crop_num = 0

        self.font = QFont()
        self.font.setFamily("Arial")  # 括号里可以设置成自己想要的其它字体
        self.font.setPointSize(21)    #字体的字号

        self.tab1 = QWidget()   #添加
        self.addTab(self.tab1, "情感分析")
        self.tab1UI()  #窗口设置
        self.resize(960, 650)

    def tab1UI(self):   #设置
        self.title=QLabel(' 人脸情感分析')  #标题名
        self.title.setFixedSize(550, 50)    #设置窗口大小
        self.title.setFont(self.font)     #字体
        # 组件展示
        self.pictureLabel = QLabel()
        self.pictureLabel.setFixedSize(550, 550)
        self.pictureLabel.setScaledContents(True)  #将图像大小适应窗口
        self.pictureLabel.setPixmap(QPixmap('./model/initial.jpg'))

        left = QVBoxLayout()#垂直布局
        left.addWidget(self.title) #添加标题
        left.addStretch(1) #伸缩填充
        left.addWidget(self.pictureLabel) #添加图片显示

        self.pic_choose_Button = QPushButton('选择图片') #选择图片按钮
        self.pic_choose_Button.setFixedSize(100, 30) #大小
        self.pic_choose_Button.clicked.connect(self.choose_pic) #事件

        self.camera_crop_Button = QPushButton('拍摄图片') #拍摄照片
        self.camera_crop_Button.setFixedSize(100, 30)#大小
        self.camera_crop_Button.clicked.connect(self.crop_pic)#事件

        top = QHBoxLayout() #水平布局
        top.addStretch(1) #自适应填充
        top.addWidget(self.pic_choose_Button) #添加按钮
        top.addStretch(1)
        top.addWidget(self.camera_crop_Button)
        top.addStretch(1)

        self.tableWidget = QTableWidget()  #
        self.tableWidget.setFixedSize(300, 450)
        self.tableWidget.setRowCount(7) #7行
        self.tableWidget.setColumnCount(1) #1列
        self.tableWidget.setHorizontalHeaderLabels(['所占比例']) #水平标签
        self.tableWidget.setVerticalHeaderLabels( #垂直标签
            ['angry', 'disgust', 'fear', 'happiness', 'sadness', 'surprise', 'neutral'])
        self.tableWidget.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        #设置自适应伸缩
        self.tableWidget.verticalHeader().setSectionResizeMode(QHeaderView.Stretch)

        right = QVBoxLayout()#垂直布局
        right.addStretch(1)
        right.addLayout(top)
        right.addStretch(1)
        right.addWidget(self.tableWidget)
        right.addStretch(1)

        #组合
        all = QHBoxLayout()  #水平布局
        all.addLayout(left)
        all.addStretch(1)
        all.addLayout(right)
        self.tab1.setLayout(all)

    def choose_pic(self):  #选择图片函数
        self.pic_choose_Button.setEnabled(True)
        self.imagename, _ = QFileDialog.getOpenFileName(self, 'Choose photo', './images/', "Image files (*.jpg *.gif *.png)")
                                                        #文本框标题，打开路径，过滤文件
        self.pictureLabel.setPixmap(QPixmap(self.imagename))
        self.predict(self.imagename)

    def predict(self, image_path):  #预测函数预处理图片
        rgb_image = cv2.imread(image_path)
        gray_image = load_image(image_path, grayscale=True)
        gray_image = np.squeeze(gray_image)
        gray_image = gray_image.astype('uint8')
        self.predict_image(rgb_image, gray_image)

    def predict_image(self, rgb_image, gray_image): #预测及输出结果

        faces = detect_faces(face_detection, gray_image) #检测人脸的存在
        for face_coordinates in faces:
            x1, x2, y1, y2 = apply_offsets(face_coordinates, emotion_offsets)#扩充
            gray_face = gray_image[y1:y2, x1:x2]#灰度化
            try:
                gray_face = cv2.resize(gray_face, (emotion_target_size))#重新塑形
            except:
                continue
            gray_face = preprocess_input(gray_face, True)#数据归一化
            gray_face = np.expand_dims(gray_face, 0)#扩展列
            gray_face = np.expand_dims(gray_face, -1)#扩展最后一个参数
            out = emotion_classifier.predict(gray_face)[0]#预测，概率
            index = out.argmax()#返回延轴最大概率的索引值
            for i, percent in enumerate(out):#组合索引序列
                b = str(round(percent * 100, 3)) + '%'
                newItem = QTableWidgetItem(str(b))#设置
                if i == index:
                    newItem.setForeground(QBrush(QColor(255, 0, 0)))#设置画笔
                self.tableWidget.setItem(i, 0, newItem)#插入到指定行列
            emotion_text = emotion_labels[index]#标签
            color = (255, 0, 0)
            draw_bounding_box(face_coordinates, rgb_image, color)#绘画
            draw_text(face_coordinates, rgb_image, emotion_text, color, 0, -20, 1, 2)#标注
        height, width = rgb_image.shape[:2]
        if rgb_image.ndim == 3:#数组维度
            rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)#颜色空间转换，RGB
        elif rgb_image.ndim == 2:
            rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_GRAY2BGR)

        temp_image = QImage(rgb_image.flatten(), width, height, 3 * width, QImage.Format_RGB888)#构造图像
        temp_pixmap = QPixmap.fromImage(temp_image)#填充
        self.pictureLabel.setPixmap(temp_pixmap)#显示

    def crop_pic(self):  #拍照
        self.camera_crop_Button.setEnabled(True)
        self.crop_num = self.crop_num+1
        camera = cv2.VideoCapture(0)
        cv2.namedWindow('MyCamera')
        while True:
            if self.crop_num % 3 == 1:
                success, self.frame = camera.read()
                cv2.imshow('MyCamera', self.frame)#显示
                if cv2.waitKey(1) & 0xff == ord('A'):#防止bug
                    break
            elif self.crop_num % 3 == 2:
                cv2.destroyWindow('MyCamera')#销毁
                camera.release()#释放
                gray_image = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)#转化
                gray_image = np.squeeze(gray_image)
                gray_image = gray_image.astype('uint8')#转换数据类型
                self.predict_image(self.frame, gray_image)#预测
                self.crop_num = self.crop_num+1
                break
            elif self.crop_num % 3 == 0:
                break

if __name__ == "__main__":
    app = QApplication(sys.argv)
    box = VideoBox()
    box.show()
    sys.exit(app.exec_())


