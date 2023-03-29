import imageio
import numpy as np
import os
import glob
import cv2


def main():
    dir_path = '/Users/lguan/Desktop/ICRA/demo/demo/'
    fname = 'frame_0_'
    save_dir = '/Users/lguan/Desktop/ICRA/demo/final/'

    all_images = glob.glob(dir_path + "*.jpg")

    current_idx = 0
    v = []
    for i in range(53):
        img_fname = os.path.join(dir_path, fname + str(i) + '.jpg')
        if img_fname in all_images:
            rgb_img = np.ones(shape=(435, 700, 3), dtype=np.uint8) * 255
            original_img = cv2.cvtColor(cv2.imread(img_fname), cv2.COLOR_BGR2RGB)

            rgb_img[0:300, 0:650, :] = original_img[0:300, 0:650, :]
            rgb_img[330:405, 450:650] = original_img[425:500, 450:650]

            img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(save_dir, 'frame_{0}.jpg'.format(current_idx)), img)
            v.append(np.copy(rgb_img))
            current_idx += 1

    # write to video
    imageio.mimwrite("/Users/lguan/Desktop/ICRA/demo/final/pacman.mp4", v, fps=3)


if __name__ == '__main__':
    main()
