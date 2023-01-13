from object_detection import ObjectDetection
from search_bert import search
from vqa import Vqa

d = ObjectDetection()

output = d.detection("./images/1.jpg")
labels = d.get_object_labels(output)
normalized_boxes, roi_features = d.get_object_features_for_vlt5(output)

v = Vqa(normalize_boxes=normalized_boxes, roi_features=roi_features)
questions = []
with open("./data/questions.txt") as f:
    for question in f.readlines():
        questions.append(question)
answers = v.get_answer(questions)

word_list = list(set(labels + answers))
print(word_list)

search(word_list)
