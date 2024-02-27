import cv2 as cv

cascade = cv.CascadeClassifier("model/haarcascade_frontalface_default.xml")  ## 读入分类器数据

sample_image = cv.imread("images/more.jpg")  ## 图片地址

faces = cascade.detectMultiScale(sample_image, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))

for (x, y, w, h) in faces:
    cv.rectangle(sample_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

if len(faces) > 0:
    print("存在" + str(len(faces)) + "张人脸")
else:
    print("不存在人脸")

cv.imshow('img', sample_image)  ## 框出的人脸图片输出到a.png
cv.waitKey(0)
cv.destroyAllWindows()