'''
本文件用于裁切视频，并选择一个评分最高的
'''

from dynamic_crop import *
from neural_image_assessment.evaluate_mobilenet import *
import logging
import glob
import torch
import torchvision

# 设置日志级别为 WARNING 或更高级别
logging.basicConfig(level=logging.WARNING)
# 设置各种裁切比例
switchers = [(1, 1), (3, 4), (4, 3), (2, 3), (3, 2), (16, 9), (9, 16)]


def generate_video_from_images(images_path, output_path, fps=30):
    '''
    本函数用于根据文件夹下的图片生成视频
    Paramaters:
        images_path: 图片文件夹路径
        output_path: 视频输出路径
        fps: 帧率
    '''
    folder_path = images_path
    image_files = glob.glob(f"{folder_path}/*.jpg") + glob.glob(f"{folder_path}/*.png")

    frames = []

    for image_file in image_files:
        # 读取图像
        image = Image.open(image_file)
        # 将图像转换为 NumPy 数组
        frame = np.array(image)
        # 添加帧到列表
        frames.append(frame)

    frames = torch.from_numpy(np.asarray(frames))
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))
    torchvision.io.write_video(output_path, frames, fps=fps, video_codec="libx264")
    # print("OUTPUT pATH is", output_path)
    return output_path



def crop_select(ori_images_path, output_crop_path, bbx_path, model_path, video_path):

    '''
    本函数用于对裁切好的图片进行评分

    Paramaters:
        images_path: 原图片文件夹
        output_crop_path: 存放切割好图片总文件夹，其下包含子文件夹，每个子文件夹代表一种裁切方式
        bbx_path: bounding_box路径
        model_path: 评价模型路径
        output_path: 输出视频文件夹路径

    Return:
        video_output_path: 输出视频的路径
    '''

    score_dict = {}
    crop_image_dict = {}
    crop_region_dict = {}

    check_paths = [output_crop_path, video_path]

    is_crop = False

    for check_path in check_paths:
        # 获取当前工作目录
        dir_name = check_path
        current_directory = os.getcwd()
        full_path = os.path.join(current_directory, dir_name)
        # 使用 os.path.exists() 检查路径是否存在
        if not os.path.exists(full_path):
            # 如果路径不存在，使用 os.makedirs() 创建目录
            os.makedirs(full_path)
            print(f"Directory '{full_path}' created.")

    # 首先清空保存裁切好图片的文件夹
    clear_folder(output_crop_path)
    
    for switch in switchers:
        images = read_images(ori_images_path)
        bbxs = read_txt_file(bbx_path)
        frame_size = images[0].size
        switch_name = f"{switch[0]}_{switch[1]}"
        try:
            rets = dynamic_program(frame_size,bbxs,switch, N=5)
        except TypeError:
            continue
        
        is_crop = True
        crop_image_dict[switch_name] = images
        crop_region,scores = rets[0]
        # 输出裁切好的图片与差分数组
        output_crop(images, output_crop_path, crop_region, scores, switch)
        crop_region_dict[switch_name] = crop_region

    if(is_crop == False):
        raise Exception("NO CROP")
    res = evaluate(output_crop_path, model_path)

    # 按值排序字典
    score_dict.update(res)
    print(crop_image_dict.keys())
    print(crop_region_dict.keys())
    # 找到最大值对应的键
    max_key = max(score_dict, key=score_dict.get)

    # 最佳切割图片集路径
    max_crop_path = output_crop_path + f"/{max_key}"
    # 输出视频路径
    video_output_path = video_path + f"{max_key}.mp4"

    # 生成视频
    generate_video_from_images(max_crop_path, video_output_path)

    return video_output_path, score_dict

    # images_to_video(crop_image_dict[max_key], crop_region_dict[max_key], output_path)




if __name__ == "__main__":
    images_path = "ori_images"
    output_crop_path = "crop_images_output"
    bbx_path = "bounding_boxes.txt"
    model_path = "neural_image_assessment/weights/mobilenet_weights.h5"
    video_path = "result/track"

    path, score_dict = crop_select(images_path, output_crop_path, bbx_path, model_path, video_path)

    # 使用sorted函数对字典的items进行排序，按值从大到小排序
    sorted_items = sorted(score_dict.items(), key=lambda x: x[1], reverse=True)

    # 从排序后的items中提取键，形成列表
    sorted_keys = [item[0] for item in sorted_items]

    print(sorted_keys)
    print(score_dict)

    print(path)



