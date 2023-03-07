import torch


device=torch.device('cuda:2')
print(device)

from transformers import AutoImageProcessor, DetrModel
from PIL import Image
import requests
import os
import numpy as np

# url = "http://images.cocodataset.org/val2017/000000039769.jpg"
# image = Image.open(requests.get(url, stream=True).raw)


in_directory = 'amazon_imgs'
out_directory = 'amazon_imgs_feature'

image_processor = AutoImageProcessor.from_pretrained("facebook/detr-resnet-50")
model = DetrModel.from_pretrained("facebook/detr-resnet-50").to(device)


in_img_buffer = []
in_img_id_butter = []
out_img_features = {}
for filename in os.listdir(in_directory):
    f = os.path.join(in_directory, filename)
    # checking if it is a file
    if os.path.isfile(f):
        id = filename[:len(filename)-4]
        image = Image.open(f)
        in_img_buffer.append(image)
        in_img_id_butter.append(id)

    if len(in_img_buffer) == 5:
        inputs = image_processor(images=in_img_buffer, return_tensors="pt").to(device)
        outputs = model(**inputs).encoder_last_hidden_state.detach()
        print(outputs)
        for i in range(outputs.shape[0]):
            print(outputs[i::].shape)
            quit()
        np.save('features.npy',out_img_features)
        # print(outputs.encoder_last_hidden_state)
        # print(outputs.encoder_last_hidden_state.shape)
        quit()





