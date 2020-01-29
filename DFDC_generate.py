import os
import cv2
import numpy as np
import multiprocessing as mp
from data import DFDC_get_batch_data
from function import generator


def job(pid):
    img_save_path = ''
    mask_save_path = ''

    img_save_path = os.path.join(img_save_path, str(pid))
    if not os.path.exists(img_save_path):
        os.mkdir(img_save_path)

    mask_save_path = os.path.join(mask_save_path, str(pid))
    if not os.path.exists(mask_save_path):
        os.mkdir(mask_save_path)

    target_num = 1000
    data_loader = DFDC_get_batch_data()
    Gen = generator()

    for i in range(target_num):
        if i == 2:
            print('process {} starts'.format(pid))
        img_files = data_loader.get_batch(batch_size=20)
        out_imgs, out_masks = Gen.data_generate(img_files, k=10)
        img_count = 0
        for img, mask in zip(out_imgs, out_masks):
            cv2.imwrite(os.path.join(img_save_path, '{0}_{1}.jpg'.format(i, img_count)), img.astype(np.uint8))
            np.save(os.path.join(mask_save_path, '{0}_{1}.npy'.format(i, img_count)), mask)
            img_count += 1



if __name__ == '__main__':
    process_num = 20
    P = []

    for i in range(process_num):
        p = mp.Process(target=job, args=(i,))
        P.append(p)

    for i in range(process_num):
        P[i].start()

    for i in range(process_num):
        P[i].join()
        

