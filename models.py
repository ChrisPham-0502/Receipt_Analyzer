import os
import torch
import numpy as np
from ultralytics import YOLO
from transformers import AutoProcessor
from transformers import AutoModelForTokenClassification
from utils import normalize_box, unnormalize_box, draw_output, create_df
from PIL import Image, ImageDraw
from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg

class Reciept_Analyzer:
    def __init__(self,
                 processor_pretrained='microsoft/layoutlmv3-base',
                 layoutlm_pretrained=os.path.join(
                      'models', 'checkpoint'),
                 yolo_pretrained=os.path.join(
                      'models', 'best.pt'),
                 vietocr_pretrained=os.path.join(
                      'models', 'vietocr', 'vgg_seq2seq.pth')
                 ):
        
        print("Initializing processor")
        if torch.cuda.is_available():
            print("Using GPU")
        else:
            print("No GPU detected, using CPU")

        self.processor = AutoProcessor.from_pretrained(
            processor_pretrained, apply_ocr=False)
        print("Finished initializing processor")

        print("Initializing LayoutLM model")
        self.lalm_model = AutoModelForTokenClassification.from_pretrained(
            layoutlm_pretrained)
        print("Finished initializing LayoutLM model")

        if yolo_pretrained is not None:
            print("Initializing YOLO model")
            self.yolo_model = YOLO(yolo_pretrained)
            print("Finished initializing YOLO model")

        print("Initializing VietOCR model")
        config = Cfg.load_config_from_name('vgg_seq2seq')
        config['weights'] = vietocr_pretrained
        config['cnn']['pretrained']= False
        config['device'] = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.vietocr = Predictor(config)
        print("Finished initializing VietOCR model")

    def forward(self, img, output_path="output", is_save_cropped_img=False):
        input_image = Image.open(img)

        # detection with YOLOv8
        bboxes = self.yolov8_det(input_image)
        
        # sort
        sorted_bboxes = self.sort_bboxes(bboxes)
        
        # draw bbox
        image_draw = input_image.copy()
        self.draw_bbox(image_draw, sorted_bboxes, output_path)

        # crop images
        cropped_images, normalized_boxes = self.get_cropped_images(input_image, sorted_bboxes, is_save_cropped_img, output_path)
        
        # recognition with VietOCR
        texts, mapping_bbox_texts = self.ocr(cropped_images, normalized_boxes)

        # KIE with LayoutLMv3
        pred_texts, pred_label, boxes  = self.kie(input_image, texts, normalized_boxes, mapping_bbox_texts, output_path)
        
        # create dataframe
        return create_df(pred_texts, pred_label)
        

    def yolov8_det(self, img):
        return self.yolo_model.predict(source=img, conf=0.3, iou=0.1)[0].boxes.xyxy.int()

    def sort_bboxes(self, bboxes):
        bbox_list = []
        for box in bboxes:
            tlx, tly, brx, bry = map(int, box)
            bbox_list.append([tlx, tly, brx, bry])
            bbox_list.sort(key=lambda x: (x[1], x[2]))
        return bbox_list
    
    def draw_bbox(self, image_draw, bboxes, output_path):
        # draw bbox
        draw = ImageDraw.Draw(image_draw)
        for box in bboxes:
            draw.rectangle(box, outline='red', width=2)
        image_draw.save(os.path.join(output_path, 'bbox.jpg'))
        print(f"Exported image with bounding boxes to {os.path.join(output_path, 'bbox.jpg')}")

    def get_cropped_images(self, input_image, bboxes, is_save_cropped=False, output_path="output"):
        normalized_boxes = []
        cropped_images = []

        # OCR
        if is_save_cropped:
            cropped_folder = os.path.join(output_path, "cropped")
            if not os.path.exists(cropped_folder):
                os.makedirs(cropped_folder)
            i = 0
        for box in bboxes:
            tlx, tly, brx, bry = map(int, box)
            normalized_box = normalize_box(box, input_image.width, input_image.height)
            normalized_boxes.append(normalized_box)
            cropped_ = input_image.crop((tlx, tly, brx, bry))
            if is_save_cropped:
                cropped_.save(os.path.join(cropped_folder, f'cropped_{i}.jpg'))
                i += 1
            cropped_images.append(cropped_)

        return cropped_images, normalized_boxes

    def ocr(self, cropped_images, normalized_boxes):
        mapping_bbox_texts = {}
        texts = []
        for img, normalized_box in zip(cropped_images, normalized_boxes):
            result = self.vietocr.predict(img)
            text = result.strip().replace('\n', ' ')
            texts.append(text)
            mapping_bbox_texts[','.join(map(str, normalized_box))] = text
        
        return texts, mapping_bbox_texts
    
    def kie(self, img, texts, boxes, mapping_bbox_texts, output_path):
        encoding = self.processor(img, texts,
                                  boxes=boxes,
                                  return_offsets_mapping=True,
                                  return_tensors='pt',
                                  max_length=512,
                                  padding='max_length')
        offset_mapping = encoding.pop('offset_mapping')

        with torch.no_grad():
            outputs = self.lalm_model(**encoding)

        id2label = self.lalm_model.config.id2label
        logits = outputs.logits
        token_boxes = encoding.bbox.squeeze().tolist()
        offset_mapping = offset_mapping.squeeze().tolist()

        predictions = logits.argmax(-1).squeeze().tolist()
        is_subword = np.array(offset_mapping)[:, 0] != 0

        true_predictions = []
        true_boxes = []
        true_texts = []
        for idx in range(len(predictions)):
            if not is_subword[idx] and token_boxes[idx] != [0, 0, 0, 0]:
                true_predictions.append(id2label[predictions[idx]])
                true_boxes.append(unnormalize_box(
                    token_boxes[idx], img.width, img.height))
                true_texts.append(mapping_bbox_texts.get(
                    ','.join(map(str, token_boxes[idx])), ''))

        if isinstance(output_path, str):
            os.makedirs(output_path, exist_ok=True)
            img_output = draw_output(
                image=img,
                true_predictions=true_predictions,
                true_boxes=true_boxes
            )
            img_output.save(os.path.join(output_path, 'result.jpg'))
            print(f"Exported result to {os.path.join(output_path, 'result.jpg')}")
        return true_texts, true_predictions, true_boxes
