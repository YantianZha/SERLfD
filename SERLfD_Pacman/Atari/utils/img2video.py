import cv2
from colour import Color
import numpy as np

from utils.color_gen import pd_linear_gradient

red = Color("white")
colors = pd_linear_gradient(start_hex='#FFFFFF', finish_hex='#0000FF', n=256)

p_colors = {8: (0, 0, 255), -8: (255, 255, 255)}


def hex2rgb(h):
    h.lstrip('#')
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def im2predicates(img_fname, save_fname, predicate_values, utility_values):
    image = cv2.imread(img_fname)

    # font
    font = cv2.FONT_HERSHEY_TRIPLEX
    org = (500, 450)
    # fontScale
    fontScale = 0.5
    # Red color in BGR
    color = (0, 0, 0)
    # Line thickness of 2 px
    thickness = 1

    predicates = ["is_ghost_nearby", "is_eat_capsule"]

    top = org[1]

    for p in predicates:
        util_value = int(np.clip(utility_values[p], -5, 5) / 10.0 * 255.0)

        uv_c = (colors.iloc[util_value]['r'].item(), colors.iloc[util_value]['g'].item(), colors.iloc[util_value]['b'].item())

        image = cv2.putText(image, p, tuple((org[0], top)), font, fontScale,
                            color, thickness, cv2.LINE_AA, False)

        image = cv2.rectangle(image, tuple((org[0] - 33, top - 12)), tuple((org[0] - 18, top + 4)), uv_c,
                              thickness=-1)
        if predicate_values[p] == 1:
            pred_color = (colors.iloc[255]['r'].item(), colors.iloc[255]['g'].item(), colors.iloc[255]['b'].item())
        else:
            pred_color = (255, 255, 255)
        image = cv2.rectangle(image, tuple((org[0] - 18, top - 12)), tuple((org[0] - 3, top + 4)), pred_color,
                              thickness=-1)

        top += 20

    cv2.imwrite(save_fname, image)



def main():
    pass


if __name__ == '__main__':
    pass
