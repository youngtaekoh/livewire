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
    dbgImage = None
    
class Settings():
    pointSize = 6

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
    if cvImage is None:
        return None
        
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
        
def segmentImage(image, seeds, iter, mult):
    img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    sitkImage = sitk.GetImageFromArray(img)            
    sitkImage = sitk.CurvatureFlow(sitkImage, 0.125, 5)
    sitkImage = sitk.ConfidenceConnected(sitkImage, seeds, iter, mult, 1, 255)
    img = sitk.GetArrayFromImage(sitkImage)
    #qImg = self.cvImage * img[..., np.newaxis]            
    return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB) 
                
class MeasureWidget(QWidget):
    messagePrinted = pyqtSignal(str)
    
    class ControlPanel(QWidget):
        # defines 'ComputeInflections' is pressed.
        requestComputeInflections = pyqtSignal()
        requestRepaint = pyqtSignal()
        
        def __init__(self, parent=None):
            super(MeasureWidget.ControlPanel, self).__init__(parent)        
            
            optLayout = QVBoxLayout()
            self.cbDrawInputCurve = QCheckBox("Input curves")
            self.cbDrawInputCurve.setChecked(True)
            self.cbDrawInputCurve.stateChanged.connect(self.emitRepaintRequest)
            optLayout.addWidget(self.cbDrawInputCurve)
            
            self.cbDrawPoints = QCheckBox("Click points")
            self.cbDrawPoints.setChecked(True)
            self.cbDrawPoints.stateChanged.connect(self.emitRepaintRequest)
            optLayout.addWidget(self.cbDrawPoints)
            
            # turn on/off inflection points
            self.cbDrawInflections = QCheckBox("Inflection points")
            self.cbDrawInflections.setChecked(True)
            self.cbDrawInflections.stateChanged.connect(self.emitRepaintRequest)
            optLayout.addWidget(self.cbDrawInflections)      
            
            # create iteration
            optLayout.addWidget(QLabel("Smoothing condition"))
            self.smoothing = QDoubleSpinBox(self)
            self.smoothing.setRange(0, 1000)
            self.smoothing.setSingleStep(1)
            self.smoothing.setValue(100.0)        
            self.smoothing.valueChanged.connect(self.requestComputeInflections)
            optLayout.addWidget(self.smoothing)
            
            def __setResolutionSpinBox(var):
                var.setRange(0, 100)
                var.setSingleStep(0.01)
                var.setDecimals(2)
                var.setValue(1)            
            
            self.sbXres = QDoubleSpinBox(self)
            __setResolutionSpinBox(self.sbXres)        
            lbXres = QLabel("horizontal res.", self)        
            lbXres.setBuddy(self.sbXres)
            
            self.sbYres = QDoubleSpinBox(self)
            __setResolutionSpinBox(self.sbYres)
            lbYres = QLabel("vertical res.", self)
            lbYres.setBuddy(self.sbYres)
            
            optLayout.addWidget(lbXres)
            optLayout.addWidget(self.sbXres)
            optLayout.addWidget(lbYres)
            optLayout.addWidget(self.sbYres)               
            
            self.btnComputeInflections = QPushButton("Compute Inflections")
            self.btnComputeInflections.pressed.connect(self.requestComputeInflections)
            optLayout.addWidget(self.btnComputeInflections)

            optLayout.addStretch()
            
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
            
        def getXres(self):
            return self.sbXres.value()
            
        def getYres(self):
            return self.sbYres.value()
            
        def getInterpolationParams(self):
            return self.smoothing.value()
            
        def adjustSmoothingCondition(self, m):
            ''' adjust smoothing spin box according to m '''
            min, max = (m-np.math.sqrt(2*m),m+np.math.sqrt(2*m))
            self.smoothing.setRange(0, max*1.5)
            self.smoothing.setValue((min + max)*0.5)
    
    class MyImage(QLabel):
        ''' Image control '''
        messagePrinted = pyqtSignal(str)
        pointsChanged = pyqtSignal()
        
        def __init__(self, controlPanel, parent=None):
            super(MeasureWidget.MyImage, self).__init__(parent)
            self.qImage = None            
            self.controlPanel = controlPanel
            self.resetComputed()
            
        def resetComputed(self):
            self.points = []
            self.curves = []
            self.interpCurve = None
            self.inflections = None
            self.curCrv = None
            self.lastPt = (0, 0)    
            self.numberOfPoints = 0    
            
        def setCVimage(self, cvImage):
            self.cvImage = cvImage
            self.qImage = ConvertCVImg2QImage(self.cvImage)        
            self.setFixedSize(self.qImage.size())     
            
            self.G = PrepareLiveWire(cvImage)
            self.p = np.zeros(self.G.shape, dtype=np.int32)
            
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
                    user_s = self.controlPanel.getInterpolationParams()
                    tck, u = interpolate.splprep([fit_x, fit_y], k=4, s=user_s)
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
                    painter.drawEllipse(p[0], p[1], Settings.pointSize, Settings.pointSize)
                
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
                        painter.drawEllipse(p[0], p[1], Settings.pointSize, Settings.pointSize)
            except TypeError:
                pass
            
        def paintEvent(self, paintEvent):
            if self.qImage is not None:
                painter = QPainter()
                painter.begin(self)
                painter.drawImage(0, 0, self.qImage)
                
                if self.controlPanel.isDrawingInputCurves():
                    self.drawCurves(painter)
                    
                self.drawInflectionPoints(painter)
                
                painter.end()
                
        def mousePressEvent(self, event):
            super(MeasureWidget.MyImage, self).mousePressEvent(event)
            if Qt.LeftButton == event.button():
                if Qt.ShiftModifier == event.modifiers():
                    if len(self.points) > 0:
                        cnt = LiveContour(self.lastPt, self.p)
                        s = tuple(cnt[-1])
                        self.curves.append(cnt)
                        self.numberOfPoints = self.numberOfPoints + len(cnt)
                        self.pointsChanged.emit()
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
                    cnt = self.curves.pop()        
                    self.numberOfPoints = self.numberOfPoints - len(cnt)
                    self.pointsChanged.emit()
                    s = self.points[-1]
                    mylivewire.mylivewire(self.p, s, self.G)
                except IndexError:
                    pass
                self.repaint()
                
        def getNumberOfPoints(self):
            return self.numberOfPoints
    
    def __init__(self, parent=None):
        super(MeasureWidget, self).__init__(parent)
        
        # Create measure widget
        self.control = MeasureWidget.ControlPanel(self)
        self.control.requestComputeInflections.connect(self.computeInflections)
        self.control.requestRepaint.connect(self.repaintRequested)
                                
        self.imageLabel = MeasureWidget.MyImage(self.control, self) # must be changed to MyImage class
        self.imageLabel.resize(400, 400) # initial size
        self.imageLabel.messagePrinted.connect(self.messagePrinted)
        self.imageLabel.pointsChanged.connect(self.pointsChanged)
        
        self.setLayout(QVBoxLayout())
        layout = QHBoxLayout()        
        layout.addWidget(self.imageLabel)
        layout.addWidget(self.control)
        self.layout().addLayout(layout)
        self.layout().addWidget(QLabel("Right button: add a point, Left button: remove a point"))
        
    def setCVimage(self, cvImage):
        self.imageLabel.setCVimage(cvImage)
        
    def reset(self):
        self.imageLabel.resetComputed()
        
    @pyqtSlot()
    def repaintRequested(self):
        self.imageLabel.repaint()
            
    @pyqtSlot()
    def computeInflections(self):
        self.imageLabel.computeInflections(np.array([self.control.getXres(), self.control.getYres()]))    
        
    @pyqtSlot()
    def pointsChanged(self):
        self.control.adjustSmoothingCondition(self.imageLabel.getNumberOfPoints())
        
class SegmentWidget(QWidget):
    messagePrinted = pyqtSignal(str)
    
    class ImageControl(QLabel):
        ''' Image control (segmentation) '''
        messagePrinted = pyqtSignal(str)
        seedChanged = pyqtSignal()
        
        def __init__(self, control, parent=None):
            super(SegmentWidget.ImageControl, self).__init__(parent)
            self.control = control
            self.cvImage = None
            self.qImage = None
            self.segImage = None
            self.qsegImage = None            
            self.seeds = []
                
        def setCVimage(self, cvImage):
            self.seeds = []
            self.cvImage = cvImage
            self.qImage = ConvertCVImg2QImage(cvImage)
            
            self.setFixedSize(self.qImage.size())
            self.repaint()
                        
        def createSegmentation(self):
            self.segImage = segmentImage(self.cvImage, self.seeds, 
                                         self.control.getIteration(),
                                         self.control.getMultiplier())
            kernsize = self.control.getKernelSize()
            kernel = np.ones((kernsize, kernsize),np.uint8)
            self.segImage = cv2.morphologyEx(self.segImage, cv2.MORPH_CLOSE, kernel)
            #DebugData.dbgImage = self.segImage.copy()
            #print "set dbgImage", type(DebugData.dbgImage)
            self.qsegImage = ConvertCVImg2QImage(self.segImage)

        def clearSegmentation(self):
            self.segImage = None
            self.qsegImage = None
                        
        def paintEvent(self, paintEvent):
            if self.qImage != None:
                painter = QPainter()
                painter.begin(self)
                painter.drawImage(0, 0, self.qImage)
                
                if self.qsegImage != None:
                    painter.setOpacity(self.control.getOpacity()/100.0)
                    painter.drawImage(0, 0, self.qsegImage)
                    painter.setOpacity(1.0)
                
                pen = QPen(painter.pen())
                pen.setColor(QColor(Qt.red))
                painter.setPen(pen)
                painter.setBrush(Qt.red)
            
                for p in self.seeds:
                    painter.drawEllipse(p[0], p[1], Settings.pointSize, Settings.pointSize)
                
                painter.end()

        def mousePressEvent(self, event):
            super(SegmentWidget.ImageControl, self).mousePressEvent(event)
            if Qt.LeftButton == event.button():
                s = (event.x(), event.y())
                self.seeds.append(s)
                self.createSegmentation()
                self.repaint()
            elif Qt.RightButton == event.button():
                if (len(self.seeds) > 0):
                    self.seeds.pop()
                    self.createSegmentation()
                    self.repaint()                    
    
    class ControlPanel(QWidget):
        ''' segmentation control '''
        
        showPreview = pyqtSignal()
        showOriginal = pyqtSignal()
        repaintRequested = pyqtSignal()
        segmentationRequested = pyqtSignal()
        
        def __init__(self, parent=None):
            super(SegmentWidget.ControlPanel, self).__init__(parent)
            
            optLayout = QHBoxLayout()
            
            # create multiplier
            optLayout.addWidget(QLabel("Multiplier"))
            self.multiplier = QDoubleSpinBox(self)
            self.multiplier.setRange(0.5, 6.0)
            self.multiplier.setSingleStep(0.1)
            self.multiplier.setValue(2.5)        
            self.multiplier.valueChanged.connect(self.segmentationRequested)
            optLayout.addWidget(self.multiplier)            
            
            # create iteration
            optLayout.addWidget(QLabel("Iteration"))
            self.iteration = QSpinBox(self)
            self.iteration.setRange(0, 40)
            self.iteration.setSingleStep(1)
            self.iteration.setValue(5)        
            self.iteration.valueChanged.connect(self.segmentationRequested)
            optLayout.addWidget(self.iteration)
            
            # kernel size
            optLayout.addWidget(QLabel("Kernel"))
            self.kernel = QSpinBox(self)
            self.kernel.setRange(3, 30)
            self.kernel.setSingleStep(1)
            self.kernel.setValue(5)        
            self.kernel.valueChanged.connect(self.segmentationRequested)
            optLayout.addWidget(self.kernel)
            
            # create opacity slider
            optLayout.addWidget(QLabel("Opacity"))
            self.opacity = QSlider(Qt.Horizontal, self)
            self.opacity.setRange(0, 100)
            self.opacity.setValue(50)    
            self.opacity.valueChanged.connect(self.changeOpacity)      
            optLayout.addWidget(self.opacity)
            self.opacityLabel = QLabel("{0}".format(self.getOpacity()))
            optLayout.addWidget(self.opacityLabel)
                                    
            self.setLayout(optLayout)           

            # original cv image
            self.cvImage = None
            
        @pyqtSlot()
        def changeOpacity(self):
            self.opacityLabel.setText("{0}".format(self.getOpacity()))    
            self.repaintRequested.emit()
            
        # @pyqtSlot()
        # def valueChanged(self, val):
        #     self.repaintRequested.emit()            
            
        def getOpacity(self):
            return self.opacity.value()
            
        def getMultiplier(self):
            return self.multiplier.value()
            
        def getIteration(self):
            return self.iteration.value()
            
        def getKernelSize(self):
            return self.kernel.value()
    
    def __init__(self, parent=None):
        super(SegmentWidget, self).__init__(parent)
        
        self.cvImage = None
        self.segmentedImage = None
        
        self.control = SegmentWidget.ControlPanel(self)
        self.control.repaintRequested.connect(self.repaintRequested)
        self.control.segmentationRequested.connect(self.segmentationRequested)
                
        self.imageLabel = SegmentWidget.ImageControl(self.control, self)
        self.imageLabel.resize(400, 400) # initial size
        self.imageLabel.messagePrinted.connect(self.messagePrinted)
        
        self.setLayout(QVBoxLayout())
        self.layout().addWidget(self.control)
        self.layout().addWidget(self.imageLabel)
        self.layout().addWidget(QLabel("Right button: add a point, Left button: remove a point"))
        
    def setCVimage(self, cvImage):
        self.imageLabel.clearSegmentation()
        self.imageLabel.setCVimage(cvImage)        
        
    def getSegmentationImage(self):
        return self.imageLabel.segImage
                
    @pyqtSlot()
    def segmentationRequested(self):
        self.imageLabel.createSegmentation()
        self.imageLabel.repaint()        
        
    @pyqtSlot()
    def repaintRequested(self):        
        self.imageLabel.repaint()
        

class MyDialog(QDialog):
    def __init__(self, parent=None):
        super(MyDialog, self).__init__(parent)
        
        self.setWindowTitle('livewire')
        
        layout = QVBoxLayout()    
        self.infoLabel = QLabel(self)
        self.infoLabel.setText("Livewire")
        self.measureWidget = MeasureWidget(self)     
        self.measureWidget.messagePrinted.connect(self.printMessage)   
        
        self.segmentWidget = SegmentWidget(self)
        self.segmentWidget.messagePrinted.connect(self.printMessage)   
        
        self.tabbedWidget = QTabWidget(self)
        self.tabbedWidget.addTab(self.segmentWidget, "segmentation")
        self.tabbedWidget.addTab(self.measureWidget, "measure")
        self.tabbedWidget.currentChanged.connect(self.tabIndexChanged)
        
        layout.addWidget(self.infoLabel)
        layout.addWidget(self.tabbedWidget)
        
        menu = self.createMenu()
        layout.setMenuBar(menu)
        
        layout.setSizeConstraint(QLayout.SetFixedSize)
        self.setLayout(layout)
        self.imageFilename = None
        
    @pyqtSlot(str)
    def printMessage(self, message):
        self.infoLabel.setText(message)
        
    @pyqtSlot(int)
    def tabIndexChanged(self, idx):
        if 1 == idx:            
            segImg = self.segmentWidget.getSegmentationImage()
            if segImg is not None:
                self.measureWidget.setCVimage(segImg)        
                    
    def load(self, filename):
        self.measureWidget.load(filename)
    
    def save(self, filename):
        self.measureWidget.save(filename)
        
    def loadImage(self):
        filename = QFileDialog.getOpenFileName(self, "Load image file", 
                                               "", "Images (*.tif *.png *.jpg)")
        if filename:
            cvImage = cv2.imread(filename)            
            self.segmentWidget.setCVimage(cvImage)
            self.measureWidget.reset()
            self.measureWidget.setCVimage(cvImage)
            self.adjustSize()
            self.tabbedWidget.setCurrentIndex(0)
            self.infoLabel.setText("{} is now loaded...".format(filename))
            self.imageFilename = filename            
            
    def loadCurve(self):
        if (self.imageFilename is not None):
            filename = QFileDialog.getOpenFileName(self, "Load curve file", 
                                                   "", "NUMPY files (*.npy)")
            if filename:
                try:
                    self.load(filename)                
                    self.measureWidget.imageLabel.repaint()
                except IOError:
                    QMessageBox.critical(self, "Error!", "Cannot load file");
        else:
            QMessageBox.warning(self, "Cannot load", "Load image first");
            
    def saveCurve(self):
        if (self.imageFilename is not None) and (len(self.measureWidget.imageLabel.points) > 1):
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
            
    #def closeEvent(self, event):
        # print 'saving debugging information'
        # DebugData.res = (self.sbXres.value(), self.sbYres.value())
        # 
        # fit_x = np.array([])
        # fit_y = np.array([])
        # for crv in self.imageLabel.curves:
        #     fit_x = np.hstack((fit_x, crv[:, 0]))
        #     fit_y = np.hstack((fit_y, crv[:, 1]))
        #     
        # DebugData.fit_x = fit_x
        # DebugData.fit_y = fit_y

def main():
    app = QApplication(sys.argv)
    w = MyDialog()
    w.resize(600, 400)
    w.show()
    app.exec_()
    
if __name__=="__main__":
    main()