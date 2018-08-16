# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS-IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys
import glob
import cv2
import os
import signal
from PyQt5 import QtGui, QtCore, QtWidgets, QtMultimedia, QtMultimediaWidgets
from labels import labels

NO_STATE = 0
RESIZE = 1

class QGraphicsRectItem(QtWidgets.QGraphicsRectItem):
    def __init__(self, scene, *args, **kwargs):
        self.scene = scene
        super(QGraphicsRectItem, self).__init__(*args, **kwargs)
        self.setFlags(QtWidgets.QGraphicsItem.ItemIsMovable | QtWidgets.QGraphicsItem.ItemSendsGeometryChanges)
        self.state = NO_STATE
    def mousePressEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton:
            if bool(event.modifiers() & QtCore.Qt.ControlModifier):
                tlw = QtWidgets.qApp.topLevelWidgets()
                for item in tlw:
                    if isinstance(item, MainWindow):
                        self.scene.removeItem(self)
                        event.accept()
                        return
            else:
                sp = event.scenePos()
                if QtGui.QVector2D(sp - self.sceneBoundingRect().bottomRight()).length() < 25:
                    self.state = RESIZE
                    event.accept()
                    return

        super(QGraphicsRectItem, self).mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if bool(event.modifiers() & QtCore.Qt.ControlModifier):
            event.ignore()
            return

        if (event.buttons() & QtCore.Qt.LeftButton):
            if self.state == RESIZE:
                r = self.rect()
                d = event.pos() - event.lastPos()
                r.adjust(0, 0, d.x(), d.y())
                self.setRect(r)
                event.accept()
                return
        super(QGraphicsRectItem, self).mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton:
            if bool(event.modifiers() & QtCore.Qt.ControlModifier):
                event.accept()
                return
            else:
                self.state = NO_STATE
        super(QGraphicsRectItem, self).mouseReleaseEvent(event)


class QGraphicsScene(QtWidgets.QGraphicsScene):
    def __init__(self, main_window, *args, **kwargs):
        self.main_window = main_window
        super(QGraphicsScene, self).__init__(*args, **kwargs)

    def keyPressEvent(self, event):
        if type(event) == QtGui.QKeyEvent:
            if event.key() == QtCore.Qt.Key_Right:
                if self.main_window.slider.value() < self.main_window.slider.maximum():
                    self.main_window.slider.setValue(self.main_window.slider.value()+1)
                else:
                    print("can't go after last")
            if event.key() == QtCore.Qt.Key_Left:
                if self.main_window.slider.value() > 0:
                    self.main_window.slider.setValue(self.main_window.slider.value()-1)
            event.accept()
        else:
            event.ignore()

    def mousePressEvent(self, event):
        super(QGraphicsScene, self).mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton:
            if bool(event.modifiers() & QtCore.Qt.ControlModifier):
                event.accept()
                return
            else:
                if not self.mouseGrabberItem():
                    end = event.scenePos()
                    start = event.buttonDownScenePos(QtCore.Qt.LeftButton)
                    tlw = QtWidgets.qApp.topLevelWidgets()
                    for item in tlw:
                        if isinstance(item, MainWindow):
                            label, okPressed = QtWidgets.QInputDialog.getItem(tlw[0], "Set label",
                                                             "Label:", labels, 0, False)
                            if okPressed and label != '':
                                self.addLabelRect(start, end, label)
        super(QGraphicsScene, self).mouseReleaseEvent(event)

    def addLabelRect(self, start, end, label):
        print("add label rect")
        rect = QtCore.QRectF(QtCore.QPointF(0.,0.), end-start)
        box = QGraphicsRectItem(self, rect)
        self.addItem(box)
        box.setPos(start)
        text = self.addSimpleText(label)
        text.setParentItem(box)
        text.setPos(rect.topLeft())

    def clearAll(self):
        pi = [item for item in self.items() if not isinstance(item, QtWidgets.QGraphicsPixmapItem)]
        for item in pi:
            self.removeItem(item)

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.central_widget = QtWidgets.QWidget(self)
        central_layout = QtWidgets.QVBoxLayout()
        self.central_widget.setLayout(central_layout)
        self.setCentralWidget(self.central_widget)

        self.view = QtWidgets.QGraphicsView()
        self.view.setDragMode(QtWidgets.QGraphicsView.RubberBandDrag)
        scene = QGraphicsScene(self)
        self.view.setScene(scene)

        self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider.valueChanged.connect(self.sliderChanged)

        central_layout.addWidget(self.view)
        central_layout.addWidget(self.slider)


        exitAction = QtWidgets.QAction(QtGui.QIcon('exit.png'), '&Exit', self)
        exitAction.setShortcut('Ctrl+Q')
        exitAction.setStatusTip('Exit application')
        exitAction.triggered.connect(QtWidgets.qApp.quit)
        saveLabelsAction = QtWidgets.QAction(QtGui.QIcon('saveLabels.png'), '&Save labels', self)
        saveLabelsAction.setStatusTip('SaveImage')
        saveLabelsAction.triggered.connect(self.saveLabels)

        menubar = self.menuBar()
        fileMenu = menubar.addMenu('&File')
        fileMenu.addAction(saveLabelsAction)
        fileMenu.addAction(exitAction)

        g = glob.glob('/home/dek/VID_20180601_095421738/*png')
        print(g)
        g.sort()
        self.loadImageFrames(g)

    def loadImageFrames(self, filenames=None):
        if not filenames:
            filenames = QtWidgets.QFileDialog.getOpenFileName(self, 'Open Files')[0]
        self.filenames = filenames
        self.readImageFrame()
        self.slider.setMinimum(0)
        self.slider.setValue(1617)
        self.slider.setMaximum(len(filenames)-1)
        self.slider.setTickInterval(100)

    def readImageFrame(self):
        filename = self.filenames[self.slider.value()]
        image = QtGui.QImage(filename, 'ARGB32')
        pixmap = QtGui.QPixmap(image)
        scene = self.view.scene()
        items = scene.items()
        pi = [item for item in items if isinstance(item, QtWidgets.QGraphicsPixmapItem)]
        if pi == []:
            scene.addPixmap(pixmap)
        else:
            pi[0].setPixmap(pixmap)
        labels_filename = os.path.join("labels", os.path.basename(filename) + ".labels")
        if os.path.exists(labels_filename):
            f = open(labels_filename)
            lines = f.readlines()
            filename = lines[0]
            for line in lines[1:]:
                line = line.strip()
                if line == '':
                    continue
                x, y, width, height, label = line.split(",")
                scene.addLabelRect(QtCore.QPointF(float(x), float(y)), QtCore.QPointF(float(width), float(height)), label)

    def sliderChanged(self):
        print(self.slider.value())
        self.view.scene().clearAll()
        self.readImageFrame()

    def saveLabels(self):
        index = self.slider.value()
        filename = self.filenames[self.slider.value()]
        labels_filename = os.path.join("labels", os.path.basename(filename) + ".labels")
        with open(labels_filename, "w") as f:
            f.write("#%s\n" % filename)
            for item in self.view.scene().items():
                if isinstance(item, QGraphicsRectItem):
                    p = item.pos()
                    label = item.childItems()[0].text()
                    f.write("%f,%f,%f,%f,%s\n" % (p.x(), p.y(), p.x()+item.rect().width(), p.y()+item.rect().height(), label))

if __name__ == '__main__':
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    app = QtWidgets.QApplication(sys.argv)
    widget = MainWindow()
    widget.show()
    app.exec_()
