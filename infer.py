import os
import torch as t
from utils.config import opt
from model import FasterRCNNVGG16
from tqdm import tqdm
from trainer import FasterRCNNTrainer
from PIL import Image
from torchvision import transforms as T
import numpy as np
from utils import array_tool as at
import cv2

VOC_BBOX_LABEL_NAMES = (
    'fly',
    'bike',
    'bird',
    'boat',
    'pin',
    'bus',
    'c',
    'cat',
    'chair',
    'cow',
    'table',
    'dog',
    'horse',
    'moto',
    'people',
    'plant',
    'shep',
    'sofa',
    'train',
    'tv',
)

def transform_image(image):
    image = np.transpose(image, (2, 0, 1))  

    image_tensor = t.tensor(image, dtype=t.float32)

    image_tensor = image_tensor.unsqueeze(0)

    return image_tensor

def draw_bboxes(image, boxes, labels, scores, threshold=0.5):
    label_names = list(VOC_BBOX_LABEL_NAMES) + ['bg']

    for i, box in enumerate(boxes):
        if scores[i] > threshold:
            
            label = label_names[labels[i]]
            
            cv2.rectangle(image, (int(box[1]), int(box[0])), (int(box[3]), int(box[2])), (0, 255, 0), 2)
            
            cv2.putText(image, label, (int(box[1]), int(box[0] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    return image

if __name__ == '__main__':
    faster_rcnn = FasterRCNNVGG16()
    trainer = FasterRCNNTrainer(faster_rcnn).cuda()
    trainer.load('save/fasterrcnn_09071339_0.6984935332483329')
    opt.caffe_pretrain=True
    cap = cv2.VideoCapture("misc/playing_ball.mp4")

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))


    out = cv2.VideoWriter('misc/output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))
    with tqdm(total=total_frames, desc="Playing video") as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            f = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            f = Image.fromarray(f)
            img = f.convert('RGB')
            img = np.asarray(img, dtype=np.float32)
            f.close()
            img = img.transpose((2, 0, 1))
            img = t.from_numpy(img)[None]
            
            with t.no_grad():
                _bboxes, _labels, _scores = trainer.faster_rcnn.predict(img,visualize=True)
            
            boxes = _bboxes[0]
            labels = _labels[0]
            scores = _scores[0]

            frame = draw_bboxes(frame, boxes, labels, scores)
            
            out.write(frame)
            pbar.update(1)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

