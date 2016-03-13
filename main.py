# -*- coding: utf-8 -*-
import sys
reload(sys)
sys.setdefaultencoding('utf-8')

import sip
sip.setapi('QString', 2)

import numpy as np
from scipy import interpolate
import cv2
import os
import SimpleITK as sitk

import mylivewire
from PyQt4.QtGui import *
from PyQt4.QtCore import *

class DebugData():
    ''' Global data for debugging '''
    res = None
    fit_x = None
    fit_y = None

def ImageToPhysics(pixel, res):
    ''' convert pixel value in image coordinate to physical coordinate '''
    x = pixel[0] * res[0] # + origin.x
    y = pixel[1] * res[1] # + origin.y
    return (x, y)
    
def PhysicsToImage(phy, res):
    px = int(phy[0] / res[0])
    py = int(phy[1] / res[1])
    return (px, py)    

def ConvertCVImg2QImage(cvImage):
    height, width, byteValue = cvImage.shape
    byteValue = byteValue * width
    cv2.cvtColor(cvImage, cv2.COLOR_BGR2RGB, cvImage)
    return QImage(cvImage, width, height, byteValue, QImage.Format_RGB888)    
    
def PrepareLiveWire(img):
    if len(img.shape) > 2:
        bwImg = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        bwImg = img
    ret, thresh = cv2.threshold(bwImg, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((7, 7), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    sobelX = cv2.Sobel(thresh, cv2.CV_8U, 1, 0, ksize=3)
    sobelY = cv2.Sobel(thresh, cv2.CV_8U, 0, 1, ksize=3)

    G = np.sqrt(sobelX.astype('float32')**2 + sobelY.astype('float32')**2)    
    return G
    
def LiveContour(t, pathmat):
    row, col = pathmat.shape
    n = pathmat[t]
    prev_n = -1
    ret = []
    while n != prev_n:
        prev_n = n;
        nidx = np.unravel_index(n, pathmat.shape);
        ret.append((nidx[1], nidx[0])) # Reverse for OpenCV
        n = pathmat[nidx]
    ret.pop()
    ret.reverse()
    return np.array(ret)
    
def drawCurve(painter, curve):
    try:        
        polyline = QPolygon()
        for pt in curve:
            polyline.append(QPoint(pt[0], pt[1]))
            painter.drawPolyline(polyline)                    
    except TypeError:
        pass
    
class ControlPanel(QWidget):
    # defines 'ComputeInflections' is pressed.
    requestComputeInflections = pyqtSignal()
    requestRepaint = pyqtSignal()
    
    def __init__(self, parent=None):
        super(ControlPanel, self).__init__(parent)        
        
        optLayout = QHBoxLayout()
        self.cbDrawInputCurve = QCheckBox("Input curves")
        self.cbDrawInputCurve.setChecked(True)
        self.cbDrawInputCurve.stateChanged.connect(self.emitRepaintRequest)
        self.cbDrawPoints = QCheckBox("Click points")
        self.cbDrawPoints.setChecked(True)
        self.cbDrawPoints.stateChanged.connect(self.emitRepaintRequest)
        self.cbDrawInflections = QCheckBox("Inflection points")
        self.cbDrawInflections.setChecked(True)
        self.cbDrawInflections.stateChanged.connect(self.emitRepaintRequest)
        self.btnComputeInflections = QPushButton("Compute Inflections")
        self.btnComputeInflections.pressed.connect(self.requestComputeInflections)
        optLayout.addWidget(self.cbDrawInputCurve)
        optLayout.addWidget(self.cbDrawPoints)
        optLayout.addWidget(self.cbDrawInflections)      
        optLayout.addWidget(self.btnComputeInflections)
        self.setLayout(optLayout)

    @pyqtSlot(int)        
    def emitRepaintRequest(self, int):
        self.requestRepaint.emit()
        
    def isDrawingInputCurves(self):
        return self.cbDrawInputCurve.isChecked()
        
    def isDrawingControlPoints(self):
        return self.cbDrawPoints.isChecked()
        
    def isDrawingInflections(self):
        return self.cbDrawInflections.isChecked()
            
class MyImage(QLabel):
    ''' Image control '''
    messagePrinted = pyqtSignal(str)
    pointSize = 6
    
    # define modes
    SegmentMode, LiveWireMode = range(2)
    mode = SegmentMode
    
    def __init__(self, controlPanel, parent=None):
        super(MyImage, self).__init__(parent)
        self.qImage = None
        self.points = []
        self.curves = []
        self.interpCurve = None
        self.inflections = None
        self.curCrv = None
        self.lastPt = (0, 0)
        self.controlPanel = controlPanel
        
    def resetComputed(self):
        self.points = []
        self.curves = []
        self.interpCurve = None
        self.inflections = None
        self.curCrv = None
        self.lastPt = (0, 0)        
        
    def setCVimage(self, cvImage):
        self.resetComputed()
        self.cvImage = cvImage
        self.qImage = ConvertCVImg2QImage(self.cvImage)        
        self.setFixedSize(self.qImage.size())     
        
        # set mode to segmentation mode
        self.mode = self.SegmentMode
        
    def computeInflections(self, res):
        if len(self.points) > 1:
            fit_x = np.array([])
            fit_y = np.array([])
            for crv in self.curves:
                fit_x = np.hstack((fit_x, crv[:, 0]))
                fit_y = np.hstack((fit_y, crv[:, 1]))
                
            # convert fit_x, fit_y into the physical coordinate.
            for i, (x, y) in enumerate(zip(fit_x, fit_y)):
                phy = ImageToPhysics((x,y), res)
                fit_x[i] = phy[0]
                fit_y[i] = phy[1]
            
            try:                
                tck, u = interpolate.splprep([fit_x, fit_y], k=4)
                out = interpolate.splev(u, tck)
                
                # convert interpolated curve points to Image Coordinate.
                for i, (x, y) in enumerate(zip(out[0], out[1])):
                    img = PhysicsToImage((x, y), res)
                    out[0][i] = img[0]
                    out[1][i] = img[1]
                    
                self.interpCurve = np.column_stack((out[0], out[1])).astype(np.int32)
                
                # compute length
                dxdy = res * np.diff(self.interpCurve, axis=0)
                length = np.sum( np.linalg.norm(dxdy, 2, axis=1) )
                
                unew = np.linspace(0, 1, len(u), endpoint=True)
                dx = interpolate.splev(unew, tck, der=1)
                ddx = interpolate.splev(unew, tck, der=2)
                k_sample = (dx[0]*ddx[1]-ddx[0]*dx[1])/np.linalg.norm(dx)**3
                fs = interpolate.InterpolatedUnivariateSpline(unew, k_sample, k=3)
                roots = fs.roots()      
    
                if (len(roots)>0):
                    outroots = []
                    out = interpolate.splev(roots, tck)
                    for ipt in np.transpose(out):
                        ipt = PhysicsToImage(ipt, res)
                        outroots.append(ipt)
                    roots = outroots            
                self.inflections = roots
                self.messagePrinted.emit("Inflection points: {} length: {:.2f} mm".format(len(roots), length))
            except ValueError as e:
                print e
                pass
        self.repaint()        
        
    def drawCurves(self, painter):
        painter.setPen(Qt.red)
        for crv in self.curves:                
            drawCurve(painter, crv)
            
        drawCurve(painter, self.curCrv)            
        if self.controlPanel.isDrawingControlPoints():
            pen = QPen(painter.pen())
            pen.setColor(QColor(Qt.red))
            painter.setPen(pen)
            painter.setBrush(Qt.red)
            
            for p in self.points:
                painter.drawEllipse(p[0], p[1], self.pointSize, self.pointSize)
            
    def drawInflectionPoints(self, painter):                    
        try:
            painter.setPen(Qt.green)
            drawCurve(painter, self.interpCurve)
            
            if self.controlPanel.isDrawingInflections():
                pen = QPen(painter.pen())
                pen.setColor(QColor(Qt.green))
                painter.setPen(pen)
                painter.setBrush(Qt.green)
            
                for p in self.inflections:
                    painter.drawEllipse(p[0], p[1], self.pointSize, self.pointSize)
        except TypeError:
            pass
        
    def paintEvent(self, paintEvent):
        if self.qImage != None:
            painter = QPainter()
            painter.begin(self)
            painter.drawImage(0, 0, self.qImage)
            
            if self.controlPanel.isDrawingInputCurves():
                self.drawCurves(painter)
                
            self.drawInflectionPoints(painter)
            
            painter.end()
            
    def livewireModeMousePressEvent(self, event):
        if Qt.LeftButton == event.button():
            if Qt.ShiftModifier == event.modifiers():
                if len(self.points) > 0:
                    cnt = LiveContour(self.lastPt, self.p)
                    s = tuple(cnt[-1])
                    self.curves.append(cnt)
                else:
                    s = (event.x(), event.y())
            
                self.points.append(s)
                mylivewire.mylivewire(self.p, s, self.G)
                self.curCrv = None
            else:
                self.lastPt = (event.y(), event.x())
                self.curCrv = LiveContour(self.lastPt, self.p)
            self.repaint()
        elif Qt.RightButton == event.button():
            self.interpCurve = None
            self.inflections = None            
            try:
                self.points.pop()        
                self.curves.pop()        
                s = self.points[-1]
                mylivewire.mylivewire(self.p, s, self.G)
            except IndexError:
                pass
            self.repaint()
            
    def segmentModeMousePressEvent(self, event):
        if Qt.LeftButton == event.button():
            s = (event.y(), event.x())        
            # 여기서 segmentation 수행            
            img = cv2.cvtColor(self.cvImage, cv2.COLOR_RGB2GRAY)
            sitkImage = sitk.GetImageFromArray(img)            
            sitkImage = sitk.CurvatureFlow(sitkImage, 0.125, 5)
            sitkImage = sitk.ConfidenceConnected(sitkImage, [s], 5, 2.5, 1, 255)
            img = sitk.GetArrayFromImage(sitkImage)
            qImg = self.cvImage * img[..., np.newaxis]            
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)            
            self.qImage = ConvertCVImg2QImage(qImg)
            
            # do this after segmentation
            self.G = PrepareLiveWire(img)
            self.p = np.zeros(self.G.shape, dtype=np.int32)
            self.repaint()
            
            self.mode = self.LiveWireMode;
        
    def mousePressEvent(self, event):
        super(MyImage, self).mousePressEvent(event)
        if self.LiveWireMode == self.mode:
            self.livewireModeMousePressEvent(event)
        elif self.SegmentMode == self.mode:
            self.segmentModeMousePressEvent(event)
        

class MyDialog(QDialog):
    def __setResolutionSpinBox(self, var):
        var.setRange(0, 100)
        var.setSingleStep(0.01)
        var.setDecimals(2)
        var.setValue(1)
    
    def __init__(self, parent=None):
        super(MyDialog, self).__init__(parent)
        
        self.setWindowTitle('livewire')
        
        layout = QVBoxLayout()    
        self.infoLabel = QLabel(self)
        self.infoLabel.setText("Livewire")

        self.controlPanel = ControlPanel(self)
        self.controlPanel.requestComputeInflections.connect(self.computeInflections)
        self.controlPanel.requestRepaint.connect(self.repaintRequested)
        
        self.sbXres = QDoubleSpinBox(self)
        self.__setResolutionSpinBox(self.sbXres)        
        lbXres = QLabel("horizontal res.", self)        
        lbXres.setBuddy(self.sbXres)
        
        self.sbYres = QDoubleSpinBox(self)
        self.__setResolutionSpinBox(self.sbYres)
        lbYres = QLabel("vertical res.", self)
        lbYres.setBuddy(self.sbYres)
        
        resolutionLayout = QHBoxLayout()
        resolutionLayout.addWidget(self.controlPanel)        
        resolutionLayout.addWidget(lbXres)
        resolutionLayout.addWidget(self.sbXres)
        resolutionLayout.addWidget(lbYres)
        resolutionLayout.addWidget(self.sbYres)        
                                
        self.imageLabel = MyImage(self.controlPanel, self) # must be changed to MyImage class
        self.imageLabel.resize(400, 400) # initial size
        self.imageLabel.messagePrinted.connect(self.printMessage)
        
        layout.addWidget(self.infoLabel)
        layout.addLayout(resolutionLayout)
        layout.addWidget(self.imageLabel)    
        
        menu = self.createMenu()
        layout.setMenuBar(menu)
        
        layout.setSizeConstraint(QLayout.SetFixedSize)
        self.setLayout(layout)
        self.imageFilename = None
        
    @pyqtSlot()
    def repaintRequested(self):
        self.imageLabel.repaint()

    @pyqtSlot(str)
    def printMessage(self, message):
        self.infoLabel.setText(message)
            
    @pyqtSlot()
    def computeInflections(self):
        self.imageLabel.computeInflections(np.array([self.sbXres.value(), self.sbYres.value()]))
        
    def load(self, filename):        
        self.imageLabel.points, self.imageLabel.curves, self.imageLabel.lastPt, res = np.load(filename)
        self.sbXres.setValue(res[0])
        self.sbYres.setValue(res[1])
    
    def save(self, filename):
        np.save(filename, [self.imageLabel.points, 
                           self.imageLabel.curves, 
                           self.imageLabel.lastPt, 
                           np.array([self.sbXres.value(), self.sbYres.value()])])
        
    def loadImage(self):
        filename = QFileDialog.getOpenFileName(self, "Load image file", 
                                               "", "Images (*.tif *.png *.jpg)")
        if filename:
            cvImage = cv2.imread(filename)
            self.imageLabel.setCVimage(cvImage)
            self.adjustSize()
            self.infoLabel.setText("{} is now loaded...".format(filename))
            self.imageFilename = filename            
            
    def loadCurve(self):
        if (self.imageFilename is not None):
            filename = QFileDialog.getOpenFileName(self, "Load curve file", 
                                                   "", "NUMPY files (*.npy)")
            if filename:
                try:
                    self.load(filename)                
                    self.imageLabel.repaint()
                except IOError:
                    QMessageBox.critical(self, "Error!", "Cannot load file");
        else:
            QMessageBox.warning(self, "Cannot load", "Load image first");
            
    def saveCurve(self):
        if (self.imageFilename is not None) and (len(self.imageLabel.points) > 1):
            pre, ext = os.path.splitext(self.imageFilename)
            filename = QFileDialog.getSaveFileName(self, "Save curve file", 
                                                   pre + ".npy", "NUMPY files (*.npy)")
            if filename:
                self.save(filename)
        else:
            QMessageBox.warning(self, "Cannot save", "There is no data to save.");
    
    def createMenu(self):
        menu = QMenuBar()
        #toolbar = 
        fileMenu = menu.addMenu("File")
        
        self.actionLoadImage = fileMenu.addAction(self.style().standardIcon(QStyle.SP_MediaPlay),
                                                  "Load image file...")
        self.actionLoadImage.triggered.connect(self.loadImage)

        fileMenu.addSeparator()
        self.actionLoadImage = fileMenu.addAction(self.style().standardIcon(QStyle.SP_DirOpenIcon),
                                                  "Load curves...")
        self.actionLoadImage.triggered.connect(self.loadCurve)        
        
        self.actionLoadImage = fileMenu.addAction(self.style().standardIcon(QStyle.SP_DriveFDIcon),
                                                 "Save curves...")
        self.actionLoadImage.triggered.connect(self.saveCurve)        
        
        return menu        

    def keyPressEvent(self, QKeyEvent):
        super(MyDialog, self).keyPressEvent(QKeyEvent)
        if Qt.Key_Escape == QKeyEvent.key():
            #app.exit(1)
            self.close()
            
    def closeEvent(self, event):
        print 'saving debugging information'
        DebugData.res = (self.sbXres.value(), self.sbYres.value())
        
        fit_x = np.array([])
        fit_y = np.array([])
        for crv in self.imageLabel.curves:
            fit_x = np.hstack((fit_x, crv[:, 0]))
            fit_y = np.hstack((fit_y, crv[:, 1]))
            
        DebugData.fit_x = fit_x
        DebugData.fit_y = fit_y

def main():
    app = QApplication(sys.argv)
    w = MyDialog()
    w.resize(600, 400)
    w.show()
    app.exec_()
    
if __name__=="__main__":
    main()