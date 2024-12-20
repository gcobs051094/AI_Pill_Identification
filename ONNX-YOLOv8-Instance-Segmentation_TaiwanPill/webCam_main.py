# Form implementation generated from reading ui file 'untitled.ui'
#
# Created by: PyQt6 UI code generator 6.7.1
#
# WARNING: Any manual changes made to this file will be lost when pyuic6 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt6 import QtCore, QtGui, QtWidgets
from PyQt6.QtGui import QImage, QPixmap, QTextCursor
from PyQt6.QtCore import Qt, QTimer
from collections import defaultdict

import cv2
import time
import sys
from yolo_detection import detector_classifier_load_model, connect_camera, frame_with_detections, get_frame, extract_and_save_features, delete_npy

class Ui_MainWindow(object):
    def initModel(self):
        # 模型路徑
        self.yolo_model_path = "models/10_40epoch_640_HP3.onnx"
        self.inceptionv3_model_path = "models/epoch_139_val_loss_1.2474.h5"
        self.conf_thres = 0.3
        self.iou_thres = 0.5
        self.frame_roi_size = 500
        
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("Pill Detection")
        MainWindow.resize(900, 850)
        
        self.centralwidget = QtWidgets.QWidget(parent=MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.verticalLayoutWidget = QtWidgets.QWidget(parent=self.centralwidget)
        self.verticalLayoutWidget.setGeometry(QtCore.QRect(0, 10, 440, 800))
        self.verticalLayoutWidget.setObjectName("verticalLayoutWidget")
        #左側
        self.leftLayout = QtWidgets.QVBoxLayout(self.verticalLayoutWidget)
        self.leftLayout.setContentsMargins(0, 0, 0, 0)
        self.leftLayout.setObjectName("leftLayout")
        #左上
        self.cameraView = QtWidgets.QLabel(parent=self.verticalLayoutWidget)
        self.cameraView.setFixedSize(400, 400)
        self.cameraView.setScaledContents(True)
        self.cameraView.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.cameraView.setObjectName("cameraView")
        self.leftLayout.addWidget(self.cameraView)
        #左下
        self.processedView = QtWidgets.QLabel(parent=self.verticalLayoutWidget)
        self.processedView.setFixedSize(400, 400)
        self.processedView.setScaledContents(True)
        self.processedView.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.processedView.setObjectName("processedView")
        self.leftLayout.addWidget(self.processedView)
        
        self.verticalLayoutWidget_2 = QtWidgets.QWidget(parent=self.centralwidget)
        self.verticalLayoutWidget_2.setGeometry(QtCore.QRect(441, 10, 440, 800))
        self.verticalLayoutWidget_2.setObjectName("verticalLayoutWidget_2")
        #右側
        self.rightLayout = QtWidgets.QVBoxLayout(self.verticalLayoutWidget_2)
        self.rightLayout.setContentsMargins(0, 0, 0, 0)
        self.rightLayout.setObjectName("rightLayout")
        #第一排-btn列
        self.btnLayout1 = QtWidgets.QHBoxLayout()
        self.btnLayout1.setSizeConstraint(QtWidgets.QLayout.SizeConstraint.SetFixedSize)
        self.btnLayout1.setObjectName("btnLayout1")
        self.btnLayout1.setSpacing(40)  # 這裡是間隔設置為 20 像素
        self.loadModelBtn = QtWidgets.QPushButton(parent=self.verticalLayoutWidget_2)
        self.loadModelBtn.setMinimumSize(QtCore.QSize(100, 50))
        self.loadModelBtn.setMaximumSize(QtCore.QSize(100, 50))
        self.loadModelBtn.setObjectName("loadModelBtn")
        self.loadModelBtn.clicked.connect(self.load_model_and_start_camera)
        self.btnLayout1.addWidget(self.loadModelBtn)
        self.imgToVectorBtn = QtWidgets.QPushButton(parent=self.verticalLayoutWidget_2)
        self.imgToVectorBtn.setMinimumSize(QtCore.QSize(100, 50))
        self.imgToVectorBtn.setMaximumSize(QtCore.QSize(100, 50))
        self.imgToVectorBtn.setObjectName("imgToVectorBtn")
        self.imgToVectorBtn.setEnabled(False)  # 初始設為無效
        self.imgToVectorBtn.clicked.connect(self.trans_img_to_vector)
        self.btnLayout1.addWidget(self.imgToVectorBtn)
        self.captureBtn = QtWidgets.QPushButton(parent=self.verticalLayoutWidget_2)
        self.captureBtn.setMinimumSize(QtCore.QSize(100, 50))
        self.captureBtn.setMaximumSize(QtCore.QSize(100, 50))
        self.captureBtn.setObjectName("captureBtn")
        self.captureBtn.clicked.connect(self.capture_frame)
        self.captureBtn.setEnabled(False)  # 初始設為無效
        self.btnLayout1.addWidget(self.captureBtn)
        self.btnLayout1.setAlignment(self.captureBtn, Qt.AlignmentFlag.AlignLeft)
        self.rightLayout.addLayout(self.btnLayout1)
        #第二排-btn列
        self.btnLayout2 = QtWidgets.QHBoxLayout()
        self.btnLayout2.setSizeConstraint(QtWidgets.QLayout.SizeConstraint.SetFixedSize)
        self.btnLayout2.setObjectName("btnLayout2")
        self.btnLayout2.setSpacing(40)  # 這裡是間隔設置為 20 像素
        self.pauseCameraBtn = QtWidgets.QPushButton(parent=self.verticalLayoutWidget_2)
        self.pauseCameraBtn.setMinimumSize(QtCore.QSize(100, 50))
        self.pauseCameraBtn.setMaximumSize(QtCore.QSize(100, 50))
        self.pauseCameraBtn.setObjectName("pauseCameraBtn")
        self.pauseCameraBtn.setEnabled(False)  # 初始設為無效
        self.pauseCameraBtn.clicked.connect(self.pause_camera)
        self.btnLayout2.addWidget(self.pauseCameraBtn)
        self.deleteVectorBtn = QtWidgets.QPushButton(parent=self.verticalLayoutWidget_2)
        self.deleteVectorBtn.setMinimumSize(QtCore.QSize(100, 50))
        self.deleteVectorBtn.setMaximumSize(QtCore.QSize(100, 50))
        self.deleteVectorBtn.setObjectName("deleteVectorBtn")
        self.deleteVectorBtn.clicked.connect(self.delete_all_vector)
        self.btnLayout2.addWidget(self.deleteVectorBtn)
        self.addBtn = QtWidgets.QPushButton(parent=self.verticalLayoutWidget_2)
        self.addBtn.setMinimumSize(QtCore.QSize(100, 50))
        self.addBtn.setMaximumSize(QtCore.QSize(100, 50))
        self.addBtn.setObjectName("addBtn")
        #self.addBtn.clicked.connect(self.load_model_and_start_camera)
        self.addBtn.setEnabled(False)  # 初始設為無效
        self.btnLayout2.addWidget(self.addBtn)
        self.btnLayout2.setAlignment(self.addBtn, Qt.AlignmentFlag.AlignLeft)
        self.rightLayout.addLayout(self.btnLayout2)
        #第三排-狀態列
        self.statusLayout = QtWidgets.QHBoxLayout()
        self.statusLayout.setSizeConstraint(QtWidgets.QLayout.SizeConstraint.SetFixedSize)
        self.statusLayout.setContentsMargins(0, -1, -1, -1)
        self.statusLayout.setSpacing(6)
        self.statusLayout.setObjectName("statusLayout")
        self.messageBox = QtWidgets.QTextEdit(parent=self.verticalLayoutWidget_2)
        self.messageBox.setMinimumSize(QtCore.QSize(440, 100))
        self.messageBox.setMaximumSize(QtCore.QSize(440, 100))
        self.messageBox.setReadOnly(True)
        self.messageBox.setObjectName("messageBox")
        self.statusLayout.addWidget(self.messageBox)
        self.rightLayout.addLayout(self.statusLayout)
        
        #第四排-影像文字列
        self.imageTextLayout = QtWidgets.QHBoxLayout()  # Create a QHBoxLayout for image and text
        self.imageTextLayout.setContentsMargins(0, 0, 0, 0)  # Remove margins
        self.imageTextLayout.setSpacing(10)  # Add spacing between items
        self.imageTextLayout.setObjectName("imageTextLayout")
        #第四排影像文字列內的物件
        self.scrollArea = QtWidgets.QScrollArea(parent=self.verticalLayoutWidget_2)
        self.scrollArea.setWidgetResizable(True)
        self.scrollArea.setObjectName("scrollArea")
        self.scrollAreaWidgetContents = QtWidgets.QWidget()
        self.scrollAreaWidgetContents.setGeometry(QtCore.QRect(0, 0, 350, 500))  # Adjust height accordingly
        self.scrollArea.setWidget(self.scrollAreaWidgetContents)
        self.scrollLayout = QtWidgets.QVBoxLayout(self.scrollAreaWidgetContents)
        self.scrollLayout.setContentsMargins(0, 0, 0, 0)
        self.scrollLayout.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop)  # Align items to the top of the layout
        self.rightLayout.addWidget(self.scrollArea)
        MainWindow.setCentralWidget(self.centralwidget)
        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)
        
        # 其他屬性
        self.camera_active = False
        self.camera_paused = False
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_camera_view)
        self.frame = None
        self.padframe = None
        self.detectframe = None
        self.cap = None
        self.yoloseg = None  # 模型變數
        self.recognotion = None  # 模型變數
        self.prev_time = 0
        self.current_time = 0
        self.fps = 0
        self.detected_objects = {}
        self.image_text_widgets = []  # Store references to QLabel and QTextEdit for later updates
        self.similarity_aggregator = defaultdict(
            lambda: {
                "total_similarity": defaultdict(float),  # 确保 total_similarity 是字典
                "count": defaultdict(int),              # 确保 count 是字典
                "image": None                            # 初始图像为 None
            }
        )

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Pill Retrieval"))
        self.cameraView.setText(_translate("MainWindow", "Camera View"))
        self.processedView.setText(_translate("MainWindow", "Detect Result"))
        self.loadModelBtn.setText(_translate("MainWindow", "Load Model"))
        self.imgToVectorBtn.setText(_translate("MainWindow", "Img to Vector"))
        self.pauseCameraBtn.setText(_translate("MainWindow", "Run/Stop Cam"))
        self.captureBtn.setText(_translate("MainWindow", "Detect"))
        self.deleteVectorBtn.setText(_translate("MainWindow", "Vector Delete"))
        self.addBtn.setText(_translate("MainWindow", " "))

    def load_model_and_start_camera(self):
        # 載入模型並啟動攝影機
        self.add_message("Loading model and starting camera...")
        self.yoloseg, self.recognotion = detector_classifier_load_model(self.yolo_model_path, self.inceptionv3_model_path, self.conf_thres, self.iou_thres)
        self.cap = connect_camera(0, 1920, 1080, 60)
        
        if self.yoloseg and self.cap:  # 確認模型和攝影機成功載入
            self.camera_active = True
            self.timer.start(30)  # 設定30ms的更新頻率
            self.add_message("Model loaded and camera started.")
            
            # 啟用其他按鈕
            self.loadModelBtn.setEnabled(False)
            self.imgToVectorBtn.setEnabled(True)  # 初始設為無效
            self.pauseCameraBtn.setEnabled(True)
            self.captureBtn.setEnabled(True)
            
        else:
            self.add_message("Failed to load model or connect to camera.")
    
    def pause_camera(self):
        """暫停或恢復攝影機畫面更新。"""
        if self.camera_active:
            if self.camera_paused:
                # 如果目前為暫停狀態，則恢復攝影機更新
                self.timer.start(30)
                self.pauseCameraBtn.setText("Pause Camera")
                self.add_message("Camera resumed.")
            else:
                # 如果目前為活動狀態，則暫停攝影機更新
                self.timer.stop()
                self.pauseCameraBtn.setText("Resume Camera")
                self.add_message("Camera paused.")
            self.camera_paused = not self.camera_paused

    def capture_frame(self):
        # 清除 info 窗口图片及相似度信息
        self.clear_image_text_widgets()
        self.clear_layout(self.imageTextLayout)
        self.add_message("Analyzing...")  # 显示分析消息

        # 检查必要条件
        if not (self.cap and self.cap.isOpened() and self.yoloseg and self.recognotion):
            self.add_message("Error: Unable to process frame.")
            return

        # 清空聚合数据
        self.similarity_aggregator.clear()
        '''
        # 处理三次不同帧
        for i in range(1):
            # 捕获不同帧
            ret, frame = self.cap.read()
            if not ret:
                self.add_message(f"Error: Unable to read frame {i+1}.")
                continue

            self.detectframe, self.rois, similarities = frame_with_detections(
                self.yoloseg, frame, self.padframe, self.recognotion
            )

            # 遍历所有的 ROI 和相似度数据
            for roi_key, similarity_list in similarities.items():
                # 获取对应的 ROI 图像
                roi_index = int(roi_key.split("_")[1]) - 1
                if roi_index < len(self.rois):
                    roi_image = self.rois[roi_index]

                    # 如果是第一次处理该 ROI，保存其图像
                    if self.similarity_aggregator[roi_key]["image"] is None:
                        self.similarity_aggregator[roi_key]["image"] = roi_image

                    # 聚合相似度数据
                    for similarity_info in similarity_list[:3]:  # 取前三个相似度最高的
                        object_name = similarity_info['image'].split('-')[0]
                        similarity_score = similarity_info['similarity']

                        self.similarity_aggregator[roi_key]["total_similarity"][object_name] += similarity_score
                        self.similarity_aggregator[roi_key]["count"][object_name] += 1

        # 处理所有 ROI 的聚合结果
        for roi_key, data in self.similarity_aggregator.items():
            # 计算每个物件的平均相似度
            avg_scores = [
                (obj_name, total / data["count"][obj_name])
                for obj_name, total in data["total_similarity"].items()
            ]

            # 按平均相似度排序并取前三名
            #cos
            top_results = sorted(avg_scores, key=lambda x: x[1], reverse=True)[:3]
            #L2
            #top_results = sorted(avg_scores, key=lambda x: x[1], reverse=False)[:3]

            # 构建显示的文字
            similarity_text = "\n".join([f"{name}: {score:.3f}" for name, score in top_results])

            # 显示影像和相似度信息
            self.add_image_text_widget(data["image"], similarity_text)
        '''
        # 捕获單帧
        ret, frame = self.cap.read()

        # 检测并获取 ROI 和相似度结果
        self.detectframe, self.rois, similarities = frame_with_detections(
            self.yoloseg, frame, self.padframe, self.recognotion
        )

        # 遍历所有的 ROI 和相似度数据
        for roi_key, similarity_list in similarities.items():
            # 获取对应的 ROI 图像
            roi_index = int(roi_key.split("_")[1]) - 1
            if roi_index < len(self.rois):
                roi_image = self.rois[roi_index]

                # 直接取前三个相似度结果
                top_results = similarity_list[:5]  # 取前三名

                # 构建显示的文字
                similarity_text = "\n".join(
                    [f"{info['image'].split('-')[0]}: {info['similarity']:.3f}" for info in top_results]
                )

                # 显示影像和相似度信息
                self.add_image_text_widget(roi_image, similarity_text)
        # 更新显示（显示最后一帧）
        rgb_image = cv2.cvtColor(self.detectframe, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        q_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)
        self.processedView.setPixmap(pixmap)
    
    #相似度計算次數
    def is_third_processing(self):
        # 这里可以通过计数器或者其他逻辑判断当前是否是第三次处理
        # 假设有一个属性 self.processing_count
        if not hasattr(self, "processing_count"):
            self.processing_count = 0
        self.processing_count += 1
        return self.processing_count == 3
        
    # Camera frame & FPS
    def update_camera_view(self):
        """Update the camera view with the given frame and detected object information."""
        self.current_time = time.time()
        
        # Ensure self.prev_time is initialized
        if not hasattr(self, 'prev_time'):
            self.prev_time = self.current_time
        
        if self.cap and self.cap.isOpened() and self.yoloseg and self.recognotion:
            self.frame, self.padframe = get_frame(self.cap, self.frame_roi_size)
            
            # FPS calculation with safety check
            time_diff = self.current_time - self.prev_time
            if time_diff > 0:  # Prevent division by zero
                self.fps = 1 / time_diff
            else:
                self.fps = 0.0  # Default FPS if time difference is zero
            
            self.prev_time = self.current_time

            # Convert frame to QImage and update the display
            rgb_image = cv2.cvtColor(self.padframe, cv2.COLOR_BGR2RGB)
            cv2.putText(rgb_image, f"FPS: {self.fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            q_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            pixmap = QPixmap.fromImage(q_image)
            self.cameraView.setPixmap(pixmap)
            
    def trans_img_to_vector(self):
        #TODO: 改成搜尋全影像資料庫
        extract_and_save_features('Taiwan_Pill_Match_Database_Matting_From_Label/', self.recognotion)
        self.add_message("Transfer Complete.")
        print("Transfer Complete.")
        self.imgToVectorBtn.setEnabled(False)
    
    def delete_all_vector(self):
        delete_npy('Taiwan_Pill_Match_Database_Matting_From_Label/')
        self.add_message("Delete Complete.")
        print("Delete Complete.")
        self.imgToVectorBtn.setEnabled(True)
    
    #添加圖片&文字
    def add_image_text_widget(self, image_array, text):
        image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
        #text = text.split("-")[1]
        
        # 將 numpy array 影像資料轉換為 QImage
        height, width, channel = image_array.shape
        bytes_per_line = 3 * width
        q_image = QImage(image_array.tobytes(), width, height, bytes_per_line, QImage.Format.Format_RGB888)

        # 將 QImage 轉為 QPixmap，並縮放大小
        pixmap = QPixmap.fromImage(q_image).scaled(90, 90, QtCore.Qt.AspectRatioMode.KeepAspectRatio)
    
        # 建立 QLabel 顯示影像
        image_label = QtWidgets.QLabel()
        image_label.setPixmap(pixmap)
        image_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)

        text_edit = QtWidgets.QLabel()
        text_edit.setText(text)
        text_edit.setAlignment(QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignVCenter | QtCore.Qt.AlignmentFlag.AlignHCenter)
        text_edit.setStyleSheet("font-size: 12px;")  # Customize text style as needed
    
        h_layout = QtWidgets.QHBoxLayout()
        h_layout.addWidget(image_label)
        h_layout.addWidget(text_edit)

        self.scrollLayout.addLayout(h_layout)
        self.image_text_widgets.append((image_label, text_edit))
        
    # Clear the scroll layout content
    def clear_image_text_widgets(self):
        # Remove all layouts and widgets from the scroll layout
        while self.scrollLayout.count():
            item = self.scrollLayout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
            elif item.layout():
                # Recursively delete any nested layouts
                self.clear_layout(item.layout())
                
        # Clear the reference list to avoid holding onto old widgets
        self.image_text_widgets.clear()

    # Helper function to clear nested layouts (if any)
    def clear_layout(self, layout):
        while layout.count():
            item = layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
            elif item.layout():
                self.clear_layout(item.layout())      
                
    def add_message(self, message):
        # 在 messageBox 中顯示訊息並自動滾動到最新訊息
        self.messageBox.append(message)
        self.messageBox.moveCursor(QTextCursor.MoveOperation.End)  # 自動滾動到最新訊息
        
        
if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    ui.initModel()
    MainWindow.show()
    sys.exit(app.exec())
