import cv2
import time
import numpy as np
import keras.utils as image
from keras.models import load_model, Model
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
import os
import imutils
from imutils import perspective
from imutils import contours

#from keras.applications.efficientnet_v2 import EfficientNetV2S, preprocess_input
#from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.applications.resnet import preprocess_input
from yoloseg import YOLOSeg

# 提供的 class_names 列表
class_names = ['capsule', 'pill_doubleround', 'pill_hexagon', 
                'pill_octagon', 'pill_oval', 'pill_pentagon',
                'pill_quadrilateral', 'pill_round', 'pill_triangle',
                'pill_waterdrop']

def detector_classifier_load_model(yolol_model_path, inceptionv3_model_path, conf_thres=0.1, iou_thres=0.1):
    """載入YOLO模型."""
    print('Loading model...')
    yoloseg = YOLOSeg(yolol_model_path, conf_thres=conf_thres, iou_thres=iou_thres)
    #model = EfficientNetV2S(weights='imagenet', include_top=False)  #1 BEST 83% 88MB inputsize=384 pre-train
    #model = InceptionV3(weights='imagenet', include_top=False)  #1 BEST 83% 88MB inputsize=299 pre-train
    model = load_model(inceptionv3_model_path)
    #layer_name = 'mixed3'  # 藥丸不適合太深/太淺的特徵，mixed3/5還行
    #layer_name = 'conv3_block4_out'  #ResNet50+Triplet
    layer_name = 'dense_6'
    model = Model(inputs=model.input, outputs=model.get_layer(layer_name).output)
    return yoloseg, model

def connect_camera(camera_id=0, width=1920, height=1080, fps=30):
    """連接攝影機並設定參數."""
    cap = cv2.VideoCapture(camera_id)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    
    actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"Requested Width: {width}, Actual Width: {actual_width}")
    print(f"Requested Height: {height}, Actual Height: {actual_height}")
    print(f"Requested FPS: {fps}, Actual FPS: {actual_fps}")
    return cap

def padding_by_zero_640(frame, height, width):
    # 依照長邊縮放，並保持比例
    if height > width:
        new_height = 640
        new_width = int(width * (640 / height))
    else:
        new_width = 640
        new_height = int(height * (640 / width))
    
    # 缩放影像
    frame_resized = cv2.resize(frame, (new_width, new_height))

    # 創建黑色背景並將縮放後的影像貼到中心
    top = (640 - new_height) // 2
    bottom = 640 - new_height - top
    left = (640 - new_width) // 2
    right = 640 - new_width - left
    
    # 將影像填充到 640x640，並保持比例
    pad = cv2.copyMakeBorder(frame_resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    
    return pad
    
def padding_by_zero_224(frame, height, width):
    # 依照長邊縮放，並保持比例
    if height > width:
        new_height = 224
        new_width = int(width * (224 / height))
    else:
        new_width = 224
        new_height = int(height * (224 / width))
    
    # 缩放影像
    frame_resized = cv2.resize(frame, (new_width, new_height))

    # 創建黑色背景並將縮放後的影像貼到中心
    top = (224 - new_height) // 2
    bottom = 224 - new_height - top
    left = (224 - new_width) // 2
    right = 224 - new_width - left
    
    # 將影像填充到 224*224，並保持比例
    pad = cv2.copyMakeBorder(frame_resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    
    return pad

def get_frame(cap, frame_roi_size):
    ret, frame = cap.read()
    if not ret:
        return None
    
    
    height, width, channels = frame.shape
    #frame = frame[int(frame_roi_size/2):height-int(frame_roi_size/2), int(frame_roi_size/2):width-int(frame_roi_size/2)]
    #height, width, channels = frame.shape
    #print(height, width)
    frame_padded = padding_by_zero_640(frame, height, width)
    
    return frame, frame_padded

# 根據 class ID 獲取物件名稱
def get_object_names_by_id(class_ids, class_names):
    # 將 class_ids 與對應的 class_names 組合成一個列表
    id_name_pairs = [(id, class_names[id]) for id in class_ids if id < len(class_names)]
    
    # 只返回名稱部分
    sorted_names = [name for _, name in id_name_pairs]
    return sorted_names
    
# 自定義 Counter 函數
def Counter(items):
    counts = {}
    for item in items:
        if item in counts:
            counts[item] += 1
        else:
            counts[item] = 1
    return counts
        
def frame_with_detections(yoloseg, frame, frame_padded, model):
    """從攝影機獲取影像並進行物件偵測，返回帶有標註的影像."""
    
    # 偵測物件
    boxes, scores, class_ids, masks = yoloseg(frame_padded)
    
    # 提取即時畫面roi和尺寸估計
    rois_draw, rois_img, longestEdges, rois_classesID = extract_roi_and_contour_size_from_masks(frame, frame_padded, boxes, class_ids, masks, 1920, 1080)
    rois_names = get_object_names_by_id(rois_classesID, class_names)
    
    # 儲存所有符合的資料夾結果
    all_matching_folders = []

    # 遍歷所有類別名稱和尺寸，生成資料夾路徑並查找相應資料夾
    for class_name, size in zip(rois_names, longestEdges):
        folder_path = get_folder_path(class_name)
        print(f"Generated Folder Path: {folder_path}")
        
        matching_folders = find_matching_folders(size, tolerance=1.0, base_path=folder_path)
        print(f"Matching Folders: {matching_folders}")
        
        all_matching_folders.append([os.path.join(folder_path, f) for f in matching_folders])

    # 輸出結果
    #print(f"All Matching Folders: {all_matching_folders}")
        
    #folder_paths = [get_folder_path(name) for name in rois_names]
    #跟影像計算相似度
    #similarities = process_images_in_folder(rois_img, all_matching_folders, model)
    #跟npy計算相似度
    similarities = calculate_similarity_for_rois(rois_img, all_matching_folders, model)
    #######
    
    # 根據 class ID 獲取物件名稱
    #sorted_names = get_object_names_by_id(class_ids, class_names)
    #object_counts = Counter(sorted_names)
    
    # 結合影像並畫出標註
    combined_img = yoloseg.draw_masks(frame_padded)

    #return combined_img, rois, object_counts, similarities
    return combined_img, rois_draw, similarities

#偵測到的影像轉成分類器吃的大小
def preprocess_roi_for_efficientnet(roi):
    roi_resized = cv2.resize(roi, (224, 224))
    roi_rgb = cv2.cvtColor(roi_resized, cv2.COLOR_BGR2RGB)  # 確保色彩順序為 RGB
    img_array = image.img_to_array(roi_rgb)  # 這裡會把 (224, 224, 3) 轉成 float32
    #img_array = img_array / 255.0  # 錯的 : 正規化到 [0, 1]
    img_array = preprocess_input(img_array)   #對的 : For ResNet50
    img_array = np.expand_dims(img_array, axis=0)  # 增加 batch 維度
    #roi_array = preprocess_input(roi_array)
    return img_array
    
#資料庫的影像轉成分類器吃的大小
def preprocess_database_for_efficientnet(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    #img_array = img_array / 255.0  # 錯的 : 正規化到 [0, 1]
    img_array = preprocess_input(img_array)   #對的 : For ResNet50
    img_array = np.expand_dims(img_array, axis=0)
    #img_array = preprocess_input(img_array)
    return img_array

#遍歷資料夾中的所有影像，並將每張影像的特徵向量儲存為 .npy 檔案。
def extract_and_save_features(folder_path, model):
    filenames = []
    
    # 使用 os.walk 遍歷資料夾及其所有子資料夾
    for root, _, files in os.walk(folder_path):
        for filename in files:
            if filename.endswith(".jpg"):  # 篩選 .jpg 檔案
                img_path = os.path.join(root, filename)
                
                if os.path.isfile(img_path):  # 確認是否為檔案
                    # 提取影像特徵
                    img_features = predict_database_with_efficientnet(img_path, model)
                    
                    # 儲存特徵向量為 .npy 檔案
                    feature_save_path = os.path.join(root, f"{os.path.splitext(filename)[0]}.npy")
                    if not os.path.exists(feature_save_path):
                        np.save(feature_save_path, img_features)
                        print(f"Features for {filename} saved to {feature_save_path}")
                    else:
                        # 如果 .npy 檔案已存在，跳過儲存
                        print(f"File {feature_save_path} already exists, skipping save.")
                    
                    filenames.append(img_path)
    
    return filenames
    
#遍歷資料夾中的所有.npy，並刪除。
def delete_npy(folder_path):
    # 遍歷資料夾中的所有檔案
    # 使用 os.walk 遍歷資料夾及其所有子資料夾
    for root, _, files in os.walk(folder_path):
        for filename in files:
            if filename.endswith(".npy"):  # 篩選出 .npy 檔案
                file_path = os.path.join(root, filename)
                
                if os.path.isfile(file_path):  # 確認是否為檔案
                    try:
                        os.remove(file_path)  # 刪除檔案
                        print(f"Deleted: {file_path}")
                    except Exception as e:
                        print(f"Failed to delete {file_path}: {e}")

#Flatten for roi
def predict_roi_with_efficientnet(roi, model):
    roi_array = preprocess_roi_for_efficientnet(roi)  
    #conv ex.3*3*512
    features = model.predict(roi_array, verbose=0).flatten()
    #已攤平 ex.1*1*1024
    #features = model.predict(roi_array, verbose=0)
    return features
    
#Flatten for database
def predict_database_with_efficientnet(img_path, model):
    img_array = preprocess_database_for_efficientnet(img_path)
    #conv ex.3*3*512
    features = model.predict(img_array, verbose=0).flatten()
    #已攤平 ex.1*1*1024
    #features = model.predict(img_array, verbose=0)
    return features

#影像資料庫cos相似度
def process_images_in_folder(rois, all_matching_folders, model):
    """
    進行相似度計算：對每個ROI和資料夾中的所有影像進行比對。
    返回每個ROI的前三個相似度最高的影像。
    """
    similarities = {}
    
    # 確保 rois 和 all_matching_folders 的數量一致
    if len(rois) != len(all_matching_folders):
        raise ValueError("rois 和 all_matching_folders 的數量不一致！")
        
    # Iterate through all detected ROIs and calculate similarity with images in the folder
    for idx, (roi, roi_folder) in enumerate(zip(rois, all_matching_folders)):
        # Get features for the current ROI
        roi_features = predict_roi_with_efficientnet(roi, model)
        
        # Compare ROI with images in the folder
        folder_similarities = {}
        for size_folder in roi_folder:
            for filename in os.listdir(size_folder):
                if filename.endswith(".jpg"):  # 只讀取 .png 檔案
                    img_path = os.path.join(size_folder, filename)
                    
                    if os.path.isfile(img_path):
                        print(f"Processing: {filename}")
                        
                        # Get features of the current image
                        img_features = predict_database_with_efficientnet(img_path, model)
                        
                        # Calculate cosine similarity between ROI and current image
                        similarity = cosine_similarity([roi_features], [img_features])[0][0]
                        folder_similarities[filename] = similarity
        
        # Sort similarities in descending order and get the top 5
        sorted_similarities = sorted(folder_similarities.items(), key=lambda x: x[1], reverse=True)
        
        # 取得最多 top 5（如果少於5則取所有結果）
        top_n = min(5, len(sorted_similarities))
        top_similarities = [{"image": name, "similarity": sim} for name, sim in sorted_similarities[:top_n]]
        
        # Store the top results for this ROI
        similarities[f"ROI_{idx+1}"] = top_similarities
    
    return similarities
    
#向量資料庫cos相似度 - 棄用
def process_vector_in_folder(rois, all_matching_folders, model):
    similarities = {}  # 儲存每個ROI對應的最相似影像
    
    # 遍歷所有資料夾，對每個資料夾中的 .npy 檔案計算相似度
    for folder_paths in all_matching_folders:
        for folder_path in folder_paths:  # 遍歷所有符合條件的資料夾
            if not os.path.exists(folder_path):
                print(f"Error: The folder path '{folder_path}' does not exist.")
                continue

            # 載入資料夾中的 .npy 檔案
            features_array = []
            filenames = []
            for filename in os.listdir(folder_path):
                if filename.endswith(".npy"):
                    npy_path = os.path.join(folder_path, filename)
                    img_features = np.load(npy_path)
                    features_array.append(img_features)
                    filenames.append(filename)
            
            features_array = np.array(features_array)

            # 對每個ROI計算相似度
            for idx, roi in enumerate(rois):
                roi_features = predict_roi_with_efficientnet(roi, model)
                
                folder_similarities = {}
                for i, img_features in enumerate(features_array):
                    similarity = cosine_similarity([roi_features], [img_features])[0][0]
                    folder_similarities[filenames[i]] = similarity

                sorted_similarities = sorted(folder_similarities.items(), key=lambda x: x[1], reverse=True)
                top_n = min(5, len(sorted_similarities))
                top_similarities = [{"image": name, "similarity": sim} for name, sim in sorted_similarities[:top_n]]

                similarities[f"ROI_{idx+1}"] = top_similarities

    return similarities

#向量資料庫cos相似度 
def calculate_similarity_for_rois(rois_img, all_matching_folders, model):
    """計算每個 ROI 與篩選資料夾中影像的相似度."""
    results = {}
    
    # 確保 rois_img 和 all_matching_folders 的數量一致
    if len(rois_img) != len(all_matching_folders):
        raise ValueError("rois_img 與 all_matching_folders 的數量不一致！")

    for idx, (roi, roi_folder) in enumerate(zip(rois_img, all_matching_folders)):
        roi_features = predict_roi_with_efficientnet(roi, model)  # ROI 的特徵向量
        roi_similarities = []  # 儲存當前 ROI 的所有匹配結果

        # 處理當前 ROI 對應的 folders
        for size_folder in roi_folder:
            for filename in os.listdir(size_folder):
                if filename.endswith(".npy"):
                    npy_path = os.path.join(size_folder, filename)
                    db_features = np.load(npy_path)  # 資料夾中影像的特徵向量
                    # 計算餘弦相似度
                    # feature_extractor
                    similarity = cosine_similarity([roi_features], [db_features])[0][0]
                    # model
                    #similarity = cosine_similarity(roi_features, db_features)[0][0]
                    # 歐基里德距離
                    #similarity = calculate_euclidean_distance(roi_features, db_features)
                    roi_similarities.append((filename, similarity))
        
        print(roi_similarities)
        print("\n ---")
        #print("\n")
        #cos
        roi_similarities = sorted(roi_similarities, key=lambda x: x[1], reverse=True)[:5]
        print(roi_similarities)
        #L2
        #roi_similarities = sorted(roi_similarities, key=lambda x: x[1], reverse=False)[:3]
        results[f"ROI_{idx+1}"] = [{"image": name, "similarity": sim} for name, sim in roi_similarities]
    
    return results
    
def extract_roi_and_contour_size_from_masks(frame, frame_padded, boxes, class_ids, masks, original_width, original_height):
    """Extract and align ROIs with background removed, using a reference object to compute sizes."""
    _frame = frame.copy()
    _frame_padded = frame_padded.copy()
    
    def restore_bbox_to_original(bbox, original_width, original_height):
        """
        將 YOLO 偵測框從填充影像 (640x640) 還原到原始影像座標。
        """
        # 計算縮放比例
        scale = 640 / original_width
        new_width = 640
        new_height = int(original_height * scale)

        # 計算填充的邊界
        top = (640 - new_height) // 2
        left = (640 - new_width) // 2

        # 還原座標
        x_min = (bbox[0] - left) / scale
        y_min = (bbox[1] - top) / scale
        x_max = (bbox[2] - left) / scale
        y_max = (bbox[3] - top) / scale

        # 確保座標不超出影像邊界
        x_min = max(0, min(original_width, x_min))
        y_min = max(0, min(original_height, y_min))
        x_max = max(0, min(original_width, x_max))
        y_max = max(0, min(original_height, y_max))

        return (int(x_min), int(y_min), int(x_max), int(y_max))
    
    def restore_mask_to_original(mask, original_width, original_height):
        """
        將 mask 從填充影像還原到原始影像尺寸。
        """
        if original_height > original_width:
            scale = 640 / original_height
            new_width = int(original_width * scale)
            new_height = 640
        else:
            scale = 640 / original_width
            new_width = 640
            new_height = int(original_height * scale)

        # 計算填充的邊界
        top = (640 - new_height) // 2
        bottom = 640 - new_height - top
        left = (640 - new_width) // 2
        right = 640 - new_width - left

        # 移除填充並還原到原始尺寸
        mask_cropped = mask[top:top+new_height, left:left+new_width]
        restored_mask = cv2.resize(mask_cropped, (original_width, original_height), interpolation=cv2.INTER_NEAREST)

        return restored_mask
    
    #判斷正方形
    def is_square(cnt):
        # 逼近多邊形
        epsilon = 0.02 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        
        # 檢查是否為四邊形
        if len(approx) == 4:
            # 計算四邊形邊長
            edges = [
                np.linalg.norm(approx[i][0] - approx[(i+1) % 4][0])
                for i in range(4)
            ]
            # 檢查邊長是否接近相等
            return max(edges) - min(edges) < 0.1 * max(edges)
        return False

    # 根據參考影像計算 pixel_per_cm
    def calculate_pixel_per_cm(reference_image):
        gray = cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY)
        equa_gray = cv2.equalizeHist(gray)
        blur = cv2.GaussianBlur(equa_gray, (9, 9), 0)
        edged = cv2.Canny(blur, 100, 150)
        edged = cv2.dilate(edged, None, iterations=1)
        edged = cv2.erode(edged, None, iterations=1)
        
        cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        
        # 排序從最左上
        cnts = sorted(cnts, key=lambda c: cv2.boundingRect(c)[0] + cv2.boundingRect(c)[1])
        #cnts = [x for x in cnts if 3000 < cv2.contourArea(x) < 5000]
        #cnts = [x for x in cnts if 3000 < cv2.contourArea(x)]
        #cv2.drawContours(frame, cnts, -1, (0,255,0), 3)
        
        cnts = [x for x in cnts if 100 < cv2.contourArea(x) and is_square(x)]
        
        
        
        if len(cnts) == 0:
            print("Can't find refernce square.")
            return reference_image
        
        print(cv2.contourArea(cnts[0]))
        ref_object = cnts[0]
        box = cv2.minAreaRect(ref_object)  #旋轉矩形
        box = cv2.boxPoints(box)
        box = np.array(box, dtype="int")
        box = perspective.order_points(box)
        (tl, tr, br, bl) = box
        
        dist_in_pixel = euclidean(tl, tr)
        dist_in_cm = 2  # Reference object size in cm
        cv2.drawContours(frame, [box.astype("int")], -1, (0, 0, 255), 2)
        
        return dist_in_pixel / dist_in_cm

    pixel_per_cm = calculate_pixel_per_cm(_frame)
    # 還原 YOLO 預測到原始影像座標
    # 還原 YOLO 預測到原始影像座標
    YoloBoxes = [restore_bbox_to_original(box, original_width, original_height) for box in boxes]
    YoloMasks = [restore_mask_to_original(mask, original_width, original_height) for mask in masks]

    # 根據 x1 座標由小到大排序
    sorted_boxes_masks = sorted(zip(YoloBoxes, YoloMasks, class_ids), key=lambda item: item[0][0])
    longest_edges = []  # 用於存儲最大的寬度或高度
    rois_draw = []
    rois_img = []
    rois_classesID = []
    
    for box, mask, class_id  in sorted_boxes_masks:
        # 取得 bounding box 座標並轉換成整數
        x1, y1, x2, y2 = map(int, box)
        
        # 確保 mask 是 uint8 類型，這是 OpenCV 需要的格式
        mask = np.uint8(mask * 255)  # 假設 mask 是 0-1 的範圍

        # 確認遮罩尺寸是否與原始框架匹配，如果不是，則縮放遮罩
        if mask.shape[:2] != _frame.shape[:2]:
            mask = cv2.resize(mask, (_frame.shape[1], _frame.shape[0]), interpolation=cv2.INTER_NEAREST)

        # 提取輪廓
        _contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # 逐一繪製每個輪廓，顏色為綠色，厚度為2
        #cv2.drawContours(_frame_padded, _contours, -1, (0, 255, 0), 2)
        
        for cnt in _contours:
            if cv2.contourArea(cnt) > 10:  # 設定最小輪廓面積過濾
                # 計算最小矩形框
                rect = cv2.minAreaRect(cnt)
                box_points = cv2.boxPoints(rect)
                box_points = np.array(box_points, dtype="int")
                
                # 確保方向一致
                width, height = rect[1]  # rect[1] 包含 (寬, 高)
                angle = rect[2]
                if width < height:  # 長邊始終為水平方向
                    angle += 90
                
                # 調整角度以確保方向不顛倒
                if angle > 90:
                    angle -= 180
                elif angle < -90:
                    angle += 180

                # 旋轉原始影像
                center = (int(rect[0][0]), int(rect[0][1]))
                rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
                rotated_frame = cv2.warpAffine(_frame, rotation_matrix, (_frame.shape[1], _frame.shape[0]))

                # 同步旋轉遮罩
                rotated_mask = cv2.warpAffine(mask, rotation_matrix, (mask.shape[1], mask.shape[0]))

                # 使用旋轉後的遮罩擷取物件
                #rotated_mask_3ch = cv2.cvtColor(rotated_mask, cv2.COLOR_GRAY2BGR)
                #沒旋轉
                mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
                #旋轉
                #isolated = cv2.bitwise_and(rotated_frame, rotated_mask_3ch)
                #沒旋轉
                isolated = cv2.bitwise_and(_frame, mask_3ch)

                # 根據旋轉後的最小邊框裁切去背的 ROI
                #x, y, w, h = cv2.boundingRect(rotated_mask)
                #沒旋轉
                x, y, w, h = cv2.boundingRect(mask)
                #mask
                roi = isolated[y:y+h, x:x+w]
                #bbox
                #roi = rotated_frame[y1:y2, x1:x2]
                
                roi = padding_by_zero_224(roi, h, w) #224*224
                rois_img.append(roi)
                
                # 計算寬高（根據 rect 的 width 和 height）
                width_cm = width / pixel_per_cm
                height_cm = height / pixel_per_cm
                longest_edge = np.max([width_cm, height_cm])
                longest_edge = round(longest_edge, 1)
                longest_edge = longest_edge * 10  #cm轉mm
                # 檢查 size 是否為無窮大或無效數值
                if longest_edge == float('inf') or longest_edge == float('-inf') or longest_edge != longest_edge:  # NaN 也會不等於自己
                    print("Error: Invalid size value (infinity or NaN)")
                    longest_edge = float('inf')  # 或者設為其他預設值
                else:
                    longest_edge = int(longest_edge)  # 將有效的 size 轉為整數
                longest_edges.append(longest_edge)
                #print("Longest_edge: " + str(longest_edge))
                
                # 在旋轉後的 ROI 上標註尺寸
                roi_center_x = w // 2
                roi_center_y = h // 2
                #印出最長邊
                cv2.putText(roi, str(longest_edge)+"mm", (roi_center_x, roi_center_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                rois_draw.append(roi)
                
                # 將 ROI 與類別 ID 存入
                rois_classesID.append(class_id)
    
    return rois_draw, rois_img, longest_edges, rois_classesID

# 根據類別名稱構建資料夾路徑
def get_folder_path(class_name):  #size此時是cm
    # 主路徑
    base_path = 'Taiwan_Pill_Match_Database_Matting_From_Label'
    
    # 如果是 pill 類別並且包含額外的子類別 (如 doubleround)
    if 'pill' in class_name:
        # pill 類別下有多層子資料夾，根據名稱解析出各層級
        path_parts = class_name.split('_')  # 例如 'pill_doubleround' 會分成 ['pill', 'doubleround']
        folder_path = f"{base_path}/{path_parts[0]}/{path_parts[1]}"  # 拼接 pill 和 doubleround 的路徑
        return folder_path
    else:
        # 如果不是 pill 類別，則直接使用類別名稱作為資料夾
        folder_path = f"{base_path}/{class_name}"
        return folder_path

#根據尺寸搜索資料夾名稱
def find_matching_folders(target_size, tolerance, base_path):
    """
    找到符合條件的資料夾名稱。
    
    :param target_size: 目標尺寸（float）
    :param tolerance: 誤差範圍（float）
    :param base_path: 資料夾的基礎路徑（字串）
    :return: 符合條件的資料夾名稱列表
    """
    matching_folders = []
    lower_bound = target_size - tolerance
    upper_bound = target_size + tolerance
    print("Target Size:" + str(target_size) + " Min Serach:" + str(lower_bound), " Max Serach:" + str(upper_bound))
    
    # 檢查 base_path 是否存在
    if not os.path.exists(base_path):
        print(f"Error: The base path '{base_path}' does not exist.")
        return matching_folders
        
    # 獲取指定路徑下的所有資料夾名稱
    folder_names = [name for name in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, name))]

    for folder in folder_names:
        # 將資料夾名稱中的 '_' 替換成 '.'，然後轉換成浮點數
        try:
            size = float(folder.replace("_", ".").replace("mm", ""))
            #print("Exist: " + str(size))
                
        except ValueError:
            # 如果資料夾名稱無法解析，跳過
            continue
        
        # 檢查尺寸是否在範圍內
        if lower_bound <= size <= upper_bound:
            matching_folders.append(folder)
    
    return matching_folders



# 計算兩點間的歐幾里得距離
def euclidean(pt1, pt2):
    return np.sqrt((pt2[0] - pt1[0]) ** 2 + (pt2[1] - pt1[1]) ** 2)

# 計算兩向量間的歐幾里得距離    
def calculate_euclidean_distance(vector1, vector2):
    return np.linalg.norm(vector1 - vector2)
    
def normalize_features(features):
    """
    將輸入的特徵向量進行L2歸一化。
    """
    return normalize(features.reshape(1, -1), norm='l2').flatten()