import torch


device=torch.device('cuda:2')
print(device)

from transformers import AutoImageProcessor, DetrModel
from PIL import Image
import requests
import os

in_directory = 'amazon_imgs'
out_directory = 'amazon_imgs_feature'

for filename in os.listdir(in_directory):
    f = os.path.join(in_directory, filename)
    # checking if it is a file
    if os.path.isfile(f):
        id = filename[:len(filename)-4]
        print(filename)
        print(id)
        quit()


url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open()

image_processor = AutoImageProcessor.from_pretrained("facebook/detr-resnet-50")
model = DetrModel.from_pretrained("facebook/detr-resnet-50").to(device)

# prepare image for the model
inputs = image_processor(images=image, return_tensors="pt").to(device)
outputs = model(**inputs)
print(outputs.encoder_last_hidden_state)



