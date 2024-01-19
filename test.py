import cv2
import numpy as np
from PIL import Image
import glob
import os
import torch
import torchvision 


def generate_video_from_frames(frames, output_path, fps=30):
    """
    Generates a video from a list of frames.
    
    Args:
        frames (list of numpy arrays): The frames to include in the video.
        output_path (str): The path to save the generated video.
        fps (int, optional): The frame rate of the output video. Defaults to 30.
    """
    
    frames = torch.from_numpy(np.asarray(frames))
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))
    torchvision.io.write_video(output_path, frames, fps=fps, video_codec="libx264")
    # print("OUTPUT pATH is", output_path)
    return output_path


folder_path = 'crop_images_output/3_2'
image_files = glob.glob(f"{folder_path}/*.jpg") + glob.glob(f"{folder_path}/*.png")

frames = []

for image_file in image_files:
    # 读取图像
    image = Image.open(image_file)

    # 将图像转换为 NumPy 数组
    frame = np.array(image)

    # 添加帧到列表
    frames.append(frame)

print(len(frames))

generate_video_from_frames(frames=frames, output_path="crop_images_output/1.mp4")