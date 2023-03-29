import cv2
import glob
import imageio
import numpy as np
from trajectory_utils import read_expert_demo
from algorithms.common.helper_functions import draw_predicates_on_img
DEMO_PATH = '/media/yz/Windows/SERL/Fetch/Push-v0/good_push_20201015_180006/'# datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
# '/home/yz/fetch_serl_' +

def im2predicates(img, num_predicates, c=-1):
    num_dims = len(img.shape)
    assert num_dims == 3 or num_dims == 4, "input is not an image"
    return img[:, c, :, :].flatten()[:num_predicates] if num_dims == 4 else img[c, :, :].flatten()[
                                                                            :num_predicates]
if __name__ == '__main__':
    num_p = 36
    demo_trajs = glob.glob(DEMO_PATH + '*.pickle')
    for demo_path in demo_trajs:
        demo_trajectory = read_expert_demo(demo_path)[0]

        # print("BBB", demo_trajectory)
        # Extract action sequence
        # action_seq = demo_trajectory[0]
        action_seq = [i[1] for i in demo_trajectory]
        state_seq = [i[0] for i in demo_trajectory] + [demo_trajectory[-1][3]]
        states = np.array([draw_predicates_on_img(s, 36, (640, 480)) for s in state_seq])
        imageio.mimwrite("demo1.mp4", states)
        for t, s in enumerate(states):
            s = cv2.resize(s, dsize=(320, 240))#(120, 90))
            cv2.imwrite('images/color_img_'+str(t)+'.jpg', s)