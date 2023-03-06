from transformers import AutoImageProcessor, DetrForObjectDetection, DetrModel
import torch
from PIL import Image
import requests

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

image_processor = AutoImageProcessor.from_pretrained("facebook/detr-resnet-50")
model = DetrModel.from_pretrained("facebook/detr-resnet-50").encoder()

inputs = image_processor(images=image, return_tensors="pt")
outputs = model(**inputs,return_dict=True)
print(outputs)



