# -*- coding: utf-8 -*-
import sys
import time

import cv2
import face_recognition
import numpy as np
from PyQt5 import QtGui, QtWidgets
from PyQt5.QtCore import pyqtSignal, pyqtSlot, Qt, QThread
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QWidget, QApplication, QLabel, QVBoxLayout
from tensorflow.keras.models import load_model

import adminpanel
import adminpassword
import checkface
import mainform
import personsaved
import wrongpassword
import db_manager as db


images = []
classNames = []
query = db.User.select()
for user in query:
    curImg = db.decode_bytes_to_img(user.img)
    images.append(curImg)
    classNames.append(user.fio)

encodeListKnown = None

model = load_model('models/masknet_v2.h5')

frame_num = 0
MIN_DISTANCE = 130
face_model = cv2.CascadeClassifier(r'haarcascade/haarcascade_frontalface_default.xml')


def addEncodingToEncodeList(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if face_recognition.face_encodings(img):
        encode = face_recognition.face_encodings(img)[0]
        encodeListKnown.append(encode)


class detected_face:
    mask_label = {
        False: 'MASK',
        True: 'NO MASK'
    }

    box_colors = {
        False: (0, 255, 0),
        True: (255, 0, 0)
    }

    def __init__(self, x, y, w, h):
        self.maskOff = True

        self.x = x
        self.y = y
        self.weight = w
        self.height = h
        self.frames_without_mask = 0

        self.center = (self.x + self.weight / 2, self.y + self.height / 2)

    def checkMaskOn(self, img):
        crop = img[self.y:self.y + self.height, self.x:self.x + self.weight]
        crop = cv2.resize(crop, (128, 128))
        crop = np.reshape(crop, [1, 128, 128, 3]) / 255.0
        self.maskOff = bool(model.predict(crop).argmax())

    def print_box(self, img):
        cv2.putText(img, self.mask_label[self.maskOff], (self.x, self.y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    self.box_colors[self.maskOff], 2)
        cv2.rectangle(img, (self.x, self.y), (self.x + self.weight, self.y + self.height),
                      self.box_colors[self.maskOff], 1)


class MainWindow(QtWidgets.QMainWindow, mainform.Ui_MainWindow):
    def __init__(self):
        global encodeListKnown
        global images
        super().__init__()

        encodeListKnown = self.findEncodings(images)

        self.setupUi(self)
        self.setWindowTitle("Главное меню")
        self.pushButton.clicked.connect(self.OnPassword)
        self.pushButton_2.clicked.connect(self.OnCheckData)

    def OnPassword(self):
        global adminpasswordwindow
        if adminpasswordwindow is None:
            adminpasswordwindow = AdminPassword(self)
        adminpasswordwindow.show()
        adminpasswordwindow.lineEdit.setText("")
        self.hide()

    def OnCheckData(self):
        global checkdatawindow
        if checkdatawindow is None:
            checkdatawindow = CheckFace(self)
        checkdatawindow.show()
        self.hide()

    def findEncodings(self, images):
        encodeList = []

        for img in images:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if face_recognition.face_encodings(img):
                encode = face_recognition.face_encodings(img)[0]
                encodeList.append(encode)
        return encodeList


class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)
    detected_face_list = []

    def __init__(self):
        super().__init__()
        self._run_flag = True

    def run(self):
        global frame_num

        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        while self._run_flag:
            ret, self.cv_img = cap.read()
            self.cv_box_img = self.cv_img.copy()
            if ret:
                frame_num += 1
                img_gray = cv2.cvtColor(self.cv_img, cv2.IMREAD_GRAYSCALE)
                faces = face_model.detectMultiScale(img_gray, scaleFactor=1.1, minNeighbors=4)

                if (frame_num % 2 == 0):

                    out_img = cv2.cvtColor(self.cv_img, cv2.COLOR_RGB2BGR)
                    self.detected_face_list = [detected_face(face[0], face[1], face[2], face[3]) for face in faces]

                    for face in self.detected_face_list:
                        face.checkMaskOn(out_img)
                        face.print_box(self.cv_box_img)
                else:
                    for face in self.detected_face_list:
                        face.print_box(self.cv_box_img)

                self.change_pixmap_signal.emit(self.cv_box_img)

        cap.release()

    def stop(self):
        self._run_flag = False
        self.wait()


class CheckFace(QWidget):
    def __init__(self, mainwindow):
        global encodeListKnown

        super().__init__()
        self.ui = checkface.Ui_Dialog()
        self.ui.setupUi(self)
        self.mainwindow = mainwindow

        self.setWindowTitle("FaceID")
        self.disply_width = self.ui.webcam_width
        self.display_height = self.ui.webcam_height
        self.image_label = QLabel(self)
        self.image_label.resize(self.disply_width, self.display_height)
        self.image_label.move((self.ui.win_width - self.disply_width) // 2 + 32, 20)

        vbox = QVBoxLayout()
        vbox.addWidget(self.image_label)
        vbox.addStretch()

        self.ui.verticalWidget.setLayout(vbox)

        self.thread = VideoThread()
        self.thread.change_pixmap_signal.connect(self.update_image)
        self.thread.start()

        self.ui.pushButton.clicked.connect(self.buttonClicked)
        self.ui.pushButton_2.clicked.connect(self.buttonClicked)

    def closeEvent(self, event):
        self.thread.stop()
        event.accept()

    @pyqtSlot(np.ndarray)
    def update_image(self, cv_img):
        qt_img = self.convert_cv_qt(cv_img)
        self.image_label.setPixmap(qt_img)

    def convert_cv_qt(self, cv_img):
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(self.disply_width, self.display_height, Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)

    def buttonClicked(self):
        global classNames

        sender = self.sender()
        matched = False
        names_set = set()
        self.ui.label.setText("")

        if sender.text() == 'Пройти проверку':
            time1 = time.time()
            time2 = time.time()
            if classNames:
                while time2 - time1 < 5:
                    time2 = time.time()
                    image = self.thread.cv_img
                    imgS = cv2.resize(image, (0, 0), None, 0.25, 0.25)
                    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
                    if face_recognition.face_locations(imgS):
                        facesCurFrame = face_recognition.face_locations(imgS)
                        encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

                        matches = []
                        matchIndex = 0

                        # print(facesCurFrame, encodesCurFrame)
                        for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
                            matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
                            faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
                            matchIndex = np.argmin(faceDis)
                            names_set.add(str(classNames[matchIndex]))

                        if matches[matchIndex]:
                            matched = True
                            break

            if matched:
                names_line = 'Найдено совпадение: '
                for name in names_set:
                    names_line += name + ', '
                names_line = names_line[:-2]
                self.ui.label.setText(names_line)
            else:
                self.ui.label.setText("Лицо не распознано")

        elif sender.text() == 'Назад':
            self.mainwindow.show()
            self.ui.label.setText("")
            self.hide()


class AdminPassword(QtWidgets.QDialog, adminpassword.Ui_Dialog):
    def __init__(self, mainwindow):
        super().__init__()
        self.setupUi(self)
        self.mainwindow = mainwindow
        self.setWindowTitle("Ввод пароля")
        self.pushButton.clicked.connect(self.OnAdminPanel)
        self.pushButton_2.clicked.connect(self.OnBack)

    def OnAdminPanel(self):
        if self.lineEdit.text() == PASSWORD:
            global adminpanelwindow
            if adminpanelwindow is None:
                adminpanelwindow = AdminPanel(self.mainwindow)
            self.lineEdit.setText('')
            adminpanelwindow.show()
        else:
            global wrongpasswordwindow
            if wrongpasswordwindow is None:
                wrongpasswordwindow = WrongPassword(self)
            self.lineEdit.setText('')
            wrongpasswordwindow.show()
        self.hide()

    def OnBack(self):
        self.mainwindow.show()
        self.hide()


class AdminPanel(QWidget):
    def __init__(self, mainwindow):
        super().__init__()
        self.mainwindow = mainwindow
        self.ui = adminpanel.Ui_Dialog()
        self.ui.setupUi(self)

        self.setWindowTitle("Добавление лица")
        self.disply_width = self.ui.webcam_width
        self.display_height = self.ui.webcam_height
        self.image_label = QLabel(self)
        self.image_label.resize(self.disply_width, self.display_height)
        self.image_label.move((self.ui.win_width - self.disply_width) // 2 + 32, 20)

        vbox = QVBoxLayout()
        vbox.addWidget(self.image_label)
        vbox.addStretch()
        self.ui.verticalWidget.setLayout(vbox)

        self.thread = VideoThread()
        self.thread.change_pixmap_signal.connect(self.update_image)
        self.thread.start()

        self.ui.pushButton.clicked.connect(self.OnSavedPerson)
        self.ui.pushButton_2.clicked.connect(self.OnBack)

    def OnSavedPerson(self):
        global personsavedwindow
        global classNames
        global images

        if personsavedwindow is None:
            personsavedwindow = PersonSaved(self)
        image = self.thread.cv_img
        fio = self.ui.lineEdit.text()
        if fio == '':
            personsavedwindow.label.setText(f"Строка ФИО пустая")
        elif face_recognition.face_locations(image):
            user_id = db.add_new_user(fio, image)
            classNames.append(fio)
            images.append(image)
            addEncodingToEncodeList(image)

            personsavedwindow.label.setText(f"{fio} (id {user_id}) сохранен")
            self.ui.lineEdit.setText('')
        else:
            personsavedwindow.label.setText(f"Лицо не найдено, попробуйте еще раз")
        personsavedwindow.show()
        self.hide()

    def closeEvent(self, event):
        self.thread.stop()
        event.accept()

    @pyqtSlot(np.ndarray)
    def update_image(self, cv_img):
        qt_img = self.convert_cv_qt(cv_img)
        self.image_label.setPixmap(qt_img)

    def convert_cv_qt(self, cv_img):
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(self.disply_width, self.display_height, Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)

    def OnBack(self):
        self.mainwindow.show()
        self.hide()


class WrongPassword(QtWidgets.QDialog, wrongpassword.Ui_Dialog):
    def __init__(self, adminpassword):
        super().__init__()
        self.adminpassword = adminpassword
        self.setupUi(self)
        self.setWindowTitle("Неверный пароль")
        self.pushButton.clicked.connect(self.OnOK)

    def OnOK(self):
        self.adminpassword.show()
        self.hide()


class PersonSaved(QtWidgets.QDialog, personsaved.Ui_Dialog):
    def __init__(self, adminpanel):
        super().__init__()
        self.setupUi(self)
        self.adminpanel = adminpanel
        self.setWindowTitle("Сохранение лица")
        self.pushButton.clicked.connect(self.OnOK)

    def OnOK(self):
        self.adminpanel.show()
        self.hide()


def main():
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


adminpasswordwindow = None
adminpanelwindow = None
checkdatawindow = None
wrongpasswordwindow = None
personsavedwindow = None
PASSWORD = '1234'


if __name__ == "__main__":
    app = QApplication(sys.argv)
    main()