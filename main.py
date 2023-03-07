import torch


device=torch.device('cuda:2')
print(device)

from transformers import AutoImageProcessor, DetrModel
from PIL import Image
import requests
import os
import numpy as np

in_directory = 'amazon_imgs'
out_directory = 'amazon_imgs_feature'

image_processor = AutoImageProcessor.from_pretrained("facebook/detr-resnet-50")
model = DetrModel.from_pretrained("facebook/detr-resnet-50").to(device)

in_img_buffer = []
for filename in os.listdir(in_directory):
    f = os.path.join(in_directory, filename)
    # checking if it is a file
    if os.path.isfile(f):
        id = filename[:len(filename)-4]
        image = Image.open(f)
        print(image)
        quit()
        in_img_buffer.append(image)

    if len(in_img_buffer) == 10:
        images = torch.FloatTensor(in_img_buffer)
        inputs = image_processor(images=images, return_tensors="pt").to(device)


# prepare image for the model
inputs = image_processor(images=image, return_tensors="pt").to(device)
outputs = model(**inputs)
print(outputs.encoder_last_hidden_state)



