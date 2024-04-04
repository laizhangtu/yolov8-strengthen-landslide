from ultralytics import YOLO

model = YOLO('E:\\code\\yolov8\\ultralytics-main\\ultralytics-main\\ultralytics\\cfg\\models\\v8\\yolov8-gold+bif+SPFFELAN+RCSOSA.yaml').load("E:\\code\\yolov8\\ultralytics-main\\runs\detect\\train32\\weights\\best.pt")

model.train(data= r'E:\\code\\yolov8\\ultralytics-main\\ultralytics-main\\yolo-ls.yaml', workers=0, epochs=100, batch=1,optimizer="AdamW")

#v1.0 优化器变为Lion----DEL
#v1.1 加入gold-yolo模块----3.28 
#v1.2 改善Iou_loss 改为WIou_loss----3.28====暂时DEL----3.29
#v1.4 发现问题出现在数据集上，转换脚本出错，框的位置发生偏移，导致精度太低----3.30
#v1.5 重新标注完3k张数据集，于rtx4090d上训练的spd网络mAP50=0.78,于rtx2080ti上训练的gold网络mAP50=0.91
#v1.6 加入bioformer----4.2
#v1.7 c2f 换成RCSOSA并将sppf换成sppelan