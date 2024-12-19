# 藥物辨識
基於台灣藥品進行藥物外觀辨識
關鍵字: image segmentation、pill identification


## 成果
1. 當前類別1845類
    *    藥丸種類: 圓形、雙圓、橢圓、三角、四邊、五邊、六邊、八邊、水滴
	*    膠囊: 膠囊
	*    辨識流程: 偵測(分割)藥丸/膠囊 -> 尺寸估計 -> 資料庫特徵匹配
    *    訓練資料從roborflow抓取


## 自建資料庫
[Taiwan_Pill_10_for_train](https://universe.roboflow.com/doccampill/taiwan_pill_10_for_train)

## Train
1. 未上傳，Local端的Train/YOLOv8_TaiwanPill

## 問題
1. 偵測(分割)穩定度不足、高相似性藥丸易誤判
    *    圓形藥丸
    *    少部分資料庫影像模糊 / 前景背景難區分(白底白藥)
    *    導致偵測受影響(混淆/漏檢)
    *    部份四邊、六邊、雙圓跟橢圓混淆
    *    橢圓跟膠囊混淆
    *    雙圓跟膠囊混淆
    *    水滴跟橢圓混淆

2. 詳細可見20241212_KC_分類器結果_藥物分配.pptx、 藥物分配專案整理及簡單說明.docx

## 未來執行
1. 將最後版本的ResNet50(self-attention)用Triplet Loss訓練
2. 更換model為EfficientB3
