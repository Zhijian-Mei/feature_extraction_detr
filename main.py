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
for ele in os.scandir(in_directory):
    size += os.path.getsize(ele)

print(size)
counter = 0
for filename in os.listdir(in_directory):
    counter += 1
    if counter % 100 == 0:
        print(counter/size)
    f = os.path.join(in_directory, filename)
    # checking if it is a file
    if os.path.isfile(f):
        id = filename[:len(filename)-4]
        try:
            image = Image.open(f)
        except:
            continue
        print(image.size)
        quit()
        in_img_buffer.append(image)
        in_img_id_butter.append(id)

    if len(in_img_buffer) == 4:
        inputs = image_processor(images=in_img_buffer, return_tensors="pt").to(device)
        outputs = model(**inputs).encoder_last_hidden_state.detach()
        for i in range(outputs.shape[0]):
            if in_img_id_butter[i] not in out_img_features:
                out_img_features[in_img_id_butter[i]] = outputs[i,:,:].cpu().numpy()
            else:
                print('error')
                quit()
        for img in in_img_buffer:
            img.close()
        in_img_buffer = []
        in_img_id_butter = []

if len(in_img_buffer) != 0:
    inputs = image_processor(images=in_img_buffer, return_tensors="pt").to(device)
    outputs = model(**inputs).encoder_last_hidden_state.detach()
    for i in range(outputs.shape[0]):
        if in_img_id_butter[i] not in out_img_features:
            out_img_features[in_img_id_butter[i]] = outputs[i, :, :].cpu().numpy()
        else:
            print('error')
            quit()
    for img in in_img_buffer:
        img.close()
    in_img_buffer = []
    in_img_id_butter = []

np.save('imgs_features.npy',out_img_features)








