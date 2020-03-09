import pandas as pd
import numpy as np
import tensorflow as tf
from PIL import Image
import os


#def image_prepare(df,imput_path)

filename = os.listdir("D:/project/train datasets/image")

base_dir = "D:/project/train datasets/image/"

new_dir  = "D:/project/train datasets/"



for img in filename:
    im = Image.open(base_dir + img).convert('L')
    im = im.resize((482,890))
    x = 40
    y = 70
    w = 400
    h = 300
    region = im.crop((x, y, x+w, y+h))
    region = region.resize((200,150))
    region = region.transpose(Image.FLIP_TOP_BOTTOM)
    region.save(new_dir + img)
    test_input=(np.array(region)//255)
    im_data=pd.DataFrame(test_input)
    img=os.path.splitext(img)[0]
    img.upper()
    im_data.to_csv(new_dir + img + '.csv')





