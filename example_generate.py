import os
import cv2
import numpy as np
from function import generator


if __name__ == '__main__':
    img_dir = 'images/original'

    Gen = generator()

    img_files = os.scandir(img_dir)
    img_files = [x.path for x in img_files]


    out_imgs , out_masks = Gen.data_generate(img_files[:50], k=30)


    # print(len(out_imgs), len(out_masks))
    #
    cv2.imshow('pic', out_imgs[-1].astype(np.uint8))
    cv2.imshow('mask', (out_masks[-1] * 255).astype(np.uint8))
    cv2.waitKey()

