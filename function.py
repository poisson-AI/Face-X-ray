import dlib
import time
from imgaug import augmenters as iaa
from tools import *


class generator:
    def __init__(self):
        self.FACE_POINTS = list(range(17, 68))
        self.MOUTH_POINTS = list(range(48, 68))
        self.RIGHT_BROW_POINTS = list(range(17, 22))
        self.LEFT_BROW_POINTS = list(range(22, 27))
        self.RIGHT_EYE_POINTS = list(range(36, 42))
        self.LEFT_EYE_POINTS = list(range(42, 48))
        self.NOSE_POINTS = list(range(27, 35))
        self.JAW_POINTS = list(range(0, 17))

        self.FACE_POINTS = list(range(0, 27))

        self.ALIGN_POINTS = (
                    self.LEFT_BROW_POINTS + self.RIGHT_EYE_POINTS + self.LEFT_EYE_POINTS + self.RIGHT_BROW_POINTS + self.NOSE_POINTS + self.MOUTH_POINTS)


        self.r1 = [self.LEFT_EYE_POINTS + self.RIGHT_EYE_POINTS + self.LEFT_BROW_POINTS + self.RIGHT_BROW_POINTS,
                          self.NOSE_POINTS + self.MOUTH_POINTS]


        def t1(images, random_state, parents, hooks):
            T = []
            iter = np.random.randint(1, 3)
            for i in range(iter):
                T.append(iaa.MinPooling(kernel_size=5, keep_size=True))
                T.append(iaa.GaussianBlur(sigma=3.))
            if np.random.rand() > 0.5:
                if np.random.rand() > 0.5:
                    T.append(iaa.MedianBlur(k=3))
                else:
                    T.append(iaa.AverageBlur(k=3))
                T.append(iaa.GaussianBlur(sigma=(0., 3.)))
            images = iaa.Sequential(T).augment_images(images)
            return images
        self.t1 = iaa.Lambda(func_images=t1)


        self.r2 = [self.FACE_POINTS]
        def t2(images, random_state, parents, hooks):
            iter = np.random.randint(6, 9)
            T = []
            for i in range(iter):
                T.append(iaa.MinPooling(kernel_size=5, keep_size=True))
                T.append(iaa.GaussianBlur(sigma=3.))
            images = iaa.Sequential(T).augment_images(images)
            return images
        self.t2 = iaa.Lambda(func_images=t2)


        self.r3 = [self.MOUTH_POINTS]
        def t3(images, random_state, parents, hooks):
            T = []
            iter = np.random.randint(1, 3)
            for i in range(iter):
                T.append(iaa.MinPooling(kernel_size=5, keep_size=True))
                T.append(iaa.GaussianBlur(sigma=3.))
            if np.random.rand() > 0.5:
                if np.random.rand() > 0.5:
                    T.append(iaa.MedianBlur(k=3))
                else:
                    T.append(iaa.AverageBlur(k=3))
                T.append(iaa.GaussianBlur(sigma=(0., 3.)))
            images = iaa.Sequential(T).augment_images(images)
            return images

        self.t3 = iaa.Lambda(func_images=t3)


        self.r4 = [self.LEFT_EYE_POINTS + self.RIGHT_EYE_POINTS]
        def t4(images, random_state, parents, hooks):
            T = []
            iter = np.random.randint(1, 3)
            for i in range(iter):
                T.append(iaa.MinPooling(kernel_size=5, keep_size=True))
                T.append(iaa.GaussianBlur(sigma=3.))
            if np.random.rand() > 0.5:
                if np.random.rand() > 0.5:
                    T.append(iaa.MedianBlur(k=3))
                else:
                    T.append(iaa.AverageBlur(k=3))
                T.append(iaa.GaussianBlur(sigma=(0., 3.)))
            images = iaa.Sequential(T).augment_images(images)
            return images
        self.t4 = iaa.Lambda(t4)


        self.r5 = [self.NOSE_POINTS]
        def t5(images, random_state, parents, hooks):
            T = []
            iter = np.random.randint(1, 3)
            for i in range(iter):
                T.append(iaa.MinPooling(kernel_size=5, keep_size=True))
                T.append(iaa.GaussianBlur(sigma=3.))
            if np.random.rand() > 0.5:
                if np.random.rand() > 0.5:
                    T.append(iaa.MedianBlur(k=3))
                else:
                    T.append(iaa.AverageBlur(k=3))
                T.append(iaa.GaussianBlur(sigma=(0., 3.)))
            images = iaa.Sequential(T).augment_images(images)
            return images
        self.t5 = iaa.Lambda(t5)


        self.r6 = [self.LEFT_EYE_POINTS + self.RIGHT_EYE_POINTS + self.NOSE_POINTS]
        def t6(images, random_state, parents, hooks):
            T = []
            iter = np.random.randint(1, 3)
            for i in range(iter):
                T.append(iaa.MinPooling(kernel_size=5, keep_size=True))
                T.append(iaa.GaussianBlur(sigma=3.))
            if np.random.rand() > 0.5:
                if np.random.rand() > 0.5:
                    T.append(iaa.MedianBlur(k=3))
                else:
                    T.append(iaa.AverageBlur(k=3))
                T.append(iaa.GaussianBlur(sigma=(0., 3.)))
            images = iaa.Sequential(T).augment_images(images)
            return images
        self.t6 = iaa.Lambda(t6)


        self.choice = [self.r1, self.r2, self.r3, self.r4, self.r5, self.r6]
        self.T_choice = [self.t1, self.t2, self.t3, self.t4, self.t5, self.t6]

        self.detector = dlib.get_frontal_face_detector()

        self.PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"
        self.predictor = dlib.shape_predictor(self.PREDICTOR_PATH)


    def get_feature_amount(self):
        return 11

    def get_color_correct_blur_frac(self):
        return 0.6


    def calculate_distance(self, pointsA, pointsB, M):
        pointsA = np.asarray(pointsA)
        pointsB = np.asarray(pointsB)
        expand = np.ones([pointsA.shape[0], 1])

        pointsB = np.concatenate([pointsB, expand], axis=-1)

        transfer_pointsB = np.dot(pointsB, np.transpose(M[:2], axes=[1, 0]))

        dis = np.mean(np.square(transfer_pointsB - pointsA))

        return dis



    def get_swap_img_and_label(self, im1, im2, landmarks1, landmarks2, M, colour_correct_blur_frac):

        target_index = np.random.randint(len(self.choice))
        target_square = self.choice[target_index]
        target_T = self.T_choice[target_index]

        points = []
        for x in target_square:
            points += x

        M = transformation_from_points(landmarks1[points],
                                                   landmarks2[points])


        amount = self.get_feature_amount()
        mask = get_face_mask(im2, landmarks2, target_square, amount)



        warped_mask = warp_im(mask, M, im1.shape)


        combined_mask = np.max([get_face_mask(im1, landmarks1, target_square, amount), warped_mask],
                               axis=0)

        warped_im2 = warp_im(im2, M, im1.shape)


        blur_amount = colour_correct_blur_frac * np.linalg.norm(
            np.mean(landmarks1[self.LEFT_EYE_POINTS], axis=0) -
            np.mean(landmarks2[self.RIGHT_EYE_POINTS], axis=0))

        warped_corrected_im2 = correct_colors(im1, warped_im2, landmarks1, landmarks2, colour_correct_blur_frac, blur_amount)


        combined_mask = self._mask_aug(combined_mask.astype(np.float32), T=target_T)

        combined_mask = np.clip(combined_mask, 0., 1.)
        # cv2.imshow('mask', (combined_mask * 255).astype(np.uint8))
        # cv2.waitKey()
        output_im = im1 * (1.0 - combined_mask) + warped_corrected_im2 * combined_mask

        combined_mask = np.mean(combined_mask, axis=-1, keepdims=False)
        label = 4 * (1.0 - combined_mask) * combined_mask

        return output_im, label

    def find_minist_k_indexes(self, np_arr, k):
        flatten_index = np.argpartition(np_arr.ravel(), k-1)[:k]
        row_index, col_index = np.unravel_index(flatten_index, np_arr.shape)

        return row_index, col_index


    def _mask_aug(self, mask, T):

        return T.augment_images([mask])[0]

    def _tail_aug(self, img):
        return img


    def data_generate(self, img_files, k):
        # start = time.time()
        # print('s', start)
        imgs = []
        landmarks = []
        #print(img_files)
        for file in img_files:
            try:
                img, landmark = read_im_and_landmarks(self.detector, self.predictor, fname=file)
                imgs.append(img)
                landmarks.append(landmark)
            except:
                continue
        if len(imgs) < k:
            k = len(imgs)
        scores = np.ones([len(imgs), len(imgs)]) * 99999
        Ms = [[None for i in range(len(imgs))] for j in range(len(imgs))]


        for i in range(len(imgs)):
            for j in range(len(imgs)):
                if i != j:
                    landmarksA = landmarks[i]
                    landmarksB = landmarks[j]

                    M = transformation_from_points(landmarksA[self.ALIGN_POINTS],
                                                   landmarksB[self.ALIGN_POINTS])

                    scores[i][j] = self.calculate_distance(landmarksA[self.ALIGN_POINTS], landmarksB[self.ALIGN_POINTS], M)

                    Ms[i][j] = M
        #print(scores)
        index_1, index_2 = self.find_minist_k_indexes(scores, k)

        out_imgs = []
        labels = []
        for i in range(k):
            M = Ms[index_1[i]][index_2[i]]
            imgA = imgs[index_1[i]]
            imgB = imgs[index_2[i]]

            landmarksA = landmarks[index_1[i]]
            landmarksB = landmarks[index_2[i]]


            out_img, label = self.get_swap_img_and_label(imgA, imgB, landmarksA, landmarksB, M, self.get_color_correct_blur_frac())
            out_imgs.append(out_img)
            labels.append(label)
        # end = time.time()
        # print('e', end)
        # print(end - start)
        return out_imgs, labels











