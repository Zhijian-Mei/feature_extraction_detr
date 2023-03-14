import torch


device=torch.device('cuda:7')
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

size = 0
for filename in os.listdir(in_directory):
    size += 1

counter = 0
for filename in os.listdir(in_directory):
    counter += 1
    if counter % 100 == 0:
        print(f'{(counter/size) * 100} %')
    f = os.path.join(in_directory, filename)
    # checking if it is a file
    if os.path.isfile(f):
        id = filename[:len(filename)-4]
        try:
            image = Image.open(f)
        except:
            continue
        if np.array(image).ndim != 3:
            continue
        # in_img_buffer.append(image)
        # in_img_id_butter.append(id)

        try:
            inputs = image_processor(images=image, return_tensors="pt").to(device)
        except:
            continue
        # outputs = model(**inputs).encoder_last_hidden_state.detach()
        outputs = model(**inputs).last_hidden_state.detach()
        if id not in out_img_features:
            out_img_features[id] = outputs.cpu().numpy()
        else:
            print('error')
            quit()
        image.close()


np.save('imgs_features.npy',out_img_features)








