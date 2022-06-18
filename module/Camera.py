import os
from PIL import Image
import cv2
import numpy as np
from PyQt5 import QtGui


class Video():
    def __init__(self, capture=0):
        self.capture = capture
        self.cam = cv2.VideoCapture(capture, cv2.CAP_DSHOW)  # mở máy ảnh
        self.currentFrame = np.array([])  # lấy khung máy ảnh hiện tại
        self.faceCascade = cv2.CascadeClassifier(
            './data/haarcascades/haarcascade_frontalface_default.xml')  # Sử dụng mô hình đào tạo khuôn mặt trực diện dựng sẵn, để nhận diện khuôn mặt
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()  # Tạo Biểu đồ Mẫu Nhị phân Cục bộ để nhận dạng khuôn mặt
        self.checkTrainerFound = os.path.isfile('trainer/trainer.yml')
        if self.checkTrainerFound == True:
            self.recognizer.read('trainer/trainer.yml')

    def quit(self):
        self.cam.release()
        cv2.destroyAllWindows()

    # Ham chụp ảnh
    def captureNextFrame(self):
        ret, readFrame = self.cam.read()
        gray = cv2.cvtColor(readFrame, cv2.COLOR_BGR2GRAY)  # tạo bức ảnh xám từ ảnh gốc
        if (ret == True):
            faces = self.faceCascade.detectMultiScale(
                gray,
                scaleFactor=1.2,
                minNeighbors=5,
                minSize=(20, 20)
            )  # phương thức detectMultiScale để phát hiện khuôn mặt trong bức ảnh xám

            for (x, y, w, h) in faces:
                cv2.rectangle(readFrame, (x, y), (x + w, y + h), (255, 0, 0),
                              2)  # Vẽ các khuôn mặt đã nhận diện được lên tấm ảnh gốc

            self.currentFrame = cv2.cvtColor(readFrame, cv2.COLOR_BGR2RGB)  # tạo ảnh xám tương tụ với các hình ccòn lại

    # Chuyen doi giua cac Frame anh
    def convertFrame(self):
        try:
            height, width = self.currentFrame.shape[:2]
            img = QtGui.QImage(self.currentFrame,
                               width,
                               height,
                               QtGui.QImage.Format_RGB888)
            img = QtGui.QPixmap.fromImage(img)
            self.previousFrame = self.currentFrame
            return img
        except:
            return None

    # Chụp ảnh và lưu ảnh vào folder dataset và chuyển ảnh về gray
    def captureFace(self, face_id):
        self.cam.set(3, 640)  # set video width
        self.cam.set(4, 480)  # set video height

        print('Đang chup ảnh khuôn mặt, hãy nhin vào máy ảnh và đợi...')
        count = 0

        while (True):
            ret, img = self.cam.read()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = self.faceCascade.detectMultiScale(gray, 1.3, 5)
            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                count += 1
                cv2.imwrite("dataset/User." + str(face_id) + '.' + str(count) + ".jpg", gray[y:y + h, x:x + w])

            # Chụp 30 ảnh mẫu và dừng video chụp ảnh
            if count >= 30:
                cv2.imwrite("image/User." + str(face_id) + ".jpg", gray[y:y + h, x:x + w])
                break

        print('Chụp khuôn mặt đã thực hiên !!!')

    # Huấn luyện ảnh sử dụng giải thuật LBPH

    def trainingFace(self):
        path = 'dataset'
        recognizer = cv2.face.LBPHFaceRecognizer_create()  # Tạo Biểu đồ Mẫu Nhị phân Cục bộ để nhận dạng khuôn mặt

        # Tạo phương thức để lấy hình ảnh và dữ liệu nhãn
        def getImagesAndLabels(path):
            imagePaths = [os.path.join(path, f) for f in os.listdir(path)]  # Nhận tất cả đường dẫn tệp
            faceSamples = []  # Khởi tạo mẫu mặt trống
            ids = []  # Khởi tạo id trống

            for imagePath in imagePaths:  # Vòng lặp tất cả đường dẫn tệp

                PIL_img = Image.open(imagePath).convert('L')  # Lấy hình ảnh và chuyển đổi nó sang thang độ xám
                img_numpy = np.array(PIL_img, 'uint8')  # PIL hình ảnh sang mảng numpy

                id = int(os.path.split(imagePath)[-1].split(".")[1])  # Lấy id hình ảnh
                faces = self.faceCascade.detectMultiScale(img_numpy)  # Lấy khuôn mặt từ các hình ảnh đào tạo

                for (x, y, w, h) in faces:  # Vòng lặp cho từng khuôn mặt, thêm vào ID tương ứng của chúng
                    faceSamples.append(img_numpy[y:y + h, x:x + w])  # Thêm hình ảnh vào các mẫu khuôn mặt
                    ids.append(id)  # Thêm ID vào ID
            return faceSamples, ids

        print('Đang đào tào khuôn mặt. vui lòng đợi...!')
        faces, ids = getImagesAndLabels(path)  # Lấy khuôn mặt và ID
        recognizer.train(faces, np.array(ids))  # Huấn luyện người mẫu bằng cách sử dụng các khuôn mặt và ID

        # Save the model into trainer/trainer.yml
        recognizer.write('trainer/trainer.yml')  # Lưu mô hình vào trainr.yml

        print("[INFO] {0} faces trained !!!".format(len(np.unique(ids))))

    # Nhận diện khuôn mặt

    def recogitionFace(self, names, defect_out):
        data = []
        name = ''
        id = ''
        confidence = 0.  # độ chính xác giữa ảnh nhận diện và ảnh huấn luyện
        minW = 0.1 * self.cam.get(3)
        minH = 0.1 * self.cam.get(4)
        ret, img = self.cam.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if (ret == True):
            faces = self.faceCascade.detectMultiScale(
                gray,
                scaleFactor=1.2,
                minNeighbors=5,
                minSize=(int(minW), int(minH)),
            )

            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                if self.checkTrainerFound == True:
                    id, confidence = self.recognizer.predict(gray[y:y + h, x:x + w])
                    # Check if confidence is less them 100 ==> is perfect match
                    if confidence < 100:
                        result_list = [d for d in names if d.get('id', '') == id]
                        name = result_list[0]['name'] if result_list else 'NULL'

                        # Only process if detection than 45%
                        if confidence <= 50:
                            defect_out.append({'id': id, 'name': name})
                        confidence = "  {0}%".format(round(100 - confidence))
                    else:
                        name = "Unknown"
                        confidence = "  {0}%".format(round(100 - confidence))
                else:
                    name = "Unknown"
                cv2.putText(img, str(name), (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(img, str(confidence), (x + 5, y + h - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 1)

            self.currentFrame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
