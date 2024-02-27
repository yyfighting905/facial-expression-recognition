from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from cnn_model import model1
from utils.data_manage import DataManager
from utils.data_manage import split_data
from utils.data_manage import preprocess_input

# 参数设置
batch_size = 32  #批训练个数
num_epochs = 1000  #训练轮数
input_shape = (48, 48, 1) #输入图像的像素
validation_split = 0.2
verbose = 1
num_classes = 7  #类别个数
patience = 50
base_path = './utils/'  #根目录

# 用于生成一个batch的图像数据，支持实时数据提升
data_generator = ImageDataGenerator(
                        featurewise_center=False,#布尔值，使图片去中心化（均值为0），按feature执行
                        featurewise_std_normalization=False,#布尔值，将输入除以数据集的标准差以完成标准化, 按feature执行
                        rotation_range=10,  #数据提升时图片随机转动的角度，整数
                        width_shift_range=0.1,#图片宽度的某个比例，数据提升时图片水平偏移的幅度，浮点数
                        height_shift_range=0.1,#图片高度的某个比例，数据提升时图片竖直偏移的幅度，浮点数
                        zoom_range=0.1,       #随机缩放幅度
                        horizontal_flip=True)   #随机水平翻转


model = model1(input_shape, num_classes)  #模型定义（输入形状，分类个数）
#定义优化器，loss function，训练过程中计算准确率,二次代价函数改为categorical_crossentropy交叉熵函数
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()   #模型保存


dataset_name = 'fer2013'  #训练集名称
print('Training data_name:', dataset_name)

# callbacks,回调
log_file_path = base_path + dataset_name + '_training.log'  # utils+fer2013
csv_logger = CSVLogger(log_file_path, append=False)
#把训练轮结果数据流到CSV文件的回调函数 append：true文件存在则增加，false覆盖存在的文件
early_stop = EarlyStopping('val_loss', patience=patience)
#当被检测的数量不再提升，则停止训练，val_loss被监测的数据，patience没有进步的训练轮数
reduce_lr = ReduceLROnPlateau('val_loss', factor=0.1, patience=int(patience/4), verbose=1)
#当标准评估停止提升时，降低学习率，val_loss被监测的数据，factor学习速率降低因素，verbose 0：安静，1：更新信息
trained_models_path = base_path + dataset_name + 'model1'
# utils+fer2013
model_names = trained_models_path + '.{epoch:02d}-{val_acc:.2f}.hdf5'
#utils+fer2013+训练轮数+验证集准确率  权值文件名
model_checkpoint = ModelCheckpoint(model_names, 'val_loss', verbose=1, save_best_only=True)
#每个训练周期后保存模型，models_names保存路径，val_loss监测数据，verbose详细信息模式，save_best_only最佳模型不覆盖
callbacks = [model_checkpoint, csv_logger, early_stop, reduce_lr]
#

# 载入数据设置
data_loader = DataManager(dataset_name, image_size=input_shape[:2])  #定义
faces_data, emotions_labels = data_loader.get_data()  #获取数据
faces_data = preprocess_input(faces_data)  #数据归一化
num_samples, num_classes = emotions_labels.shape
train_data, val_data = split_data(faces_data, emotions_labels, validation_split) #设定训练集、验证集
train_faces, train_emotions = train_data  #train_x,train_y
model.fit_generator(data_generator.flow(train_faces, train_emotions, batch_size),
                    steps_per_epoch=len(train_faces) / batch_size,
                    epochs=num_epochs, verbose=1, callbacks=callbacks,
                    validation_data=val_data)
#接收numpy数组和标签为参数，生成经过数据提升或标准化后的batch数据,并在一个无限循环中不断的返回batch数据
#train_faces样本数据，train_emotions标签,batch_size批处理个数，epoch轮数，verbose 1：输出进度条记录