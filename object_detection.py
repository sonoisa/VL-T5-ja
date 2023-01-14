import torch
from VLT5.inference.processing_image import Preprocess
from VLT5.inference.modeling_frcnn import GeneralizedRCNN
from VLT5.inference.utils import Config
import unicodedata


class ObjectDetection:
    def __init__(self):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        # 日本語訳した物体の分類ラベル
        self.obj_ids = []
        with open("./VLT5/VG/objects_vocab.txt") as f:
            for obj in f.readlines():
                obj = unicodedata.normalize("NFKC", obj)
                self.obj_ids.append(obj.split(",")[0].lower().strip())

        # Faster-RCNN読み込み
        self.frcnn_cfg = Config.from_pretrained("unc-nlp/frcnn-vg-finetuned")
        self.frcnn = GeneralizedRCNN.from_pretrained(
            "unc-nlp/frcnn-vg-finetuned", config=self.frcnn_cfg
        )
        self.frcnn.to(self.device)

    def detection(self, image_path):
        image_preprocess = Preprocess(self.frcnn_cfg)
        images, sizes, scales_yx = image_preprocess(image_path)
        images = images.to(self.device)

        output_dict = self.frcnn(
            images,
            sizes,
            scales_yx=scales_yx,
            padding="max_detections",
            max_detections=self.frcnn_cfg.max_detections,
            return_tensors="pt",
        )
        return output_dict

    def get_object_labels(self, output_dict):
        labels = []
        # 物体ラベルを追加
        for id in output_dict.get("obj_ids")[0]:
            labels.append(self.obj_ids[id])
        return labels

    def get_object_features_for_vlt5(self, output_dict):
        normalized_boxes = output_dict.get("normalized_boxes")
        roi_features = output_dict.get("roi_features")
        return normalized_boxes, roi_features
