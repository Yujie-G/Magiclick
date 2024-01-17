import numpy as np
import os
from PIL import Image
import cv2

class Before:
    def __init__(self,before,node,distance):
        self.before = before
        self.node = node
        self.distance = distance

class Node:
    def __init__(self, prop, bbx, N=10):
        self.N = N
        self.x1, self.y1, self.x2, self.y2 = prop
        self.bx1, self.by1, self.bx2, self.by2 = bbx
        self.alpha = self.get_alpha()
        self.theta = self.get_theta()
        self.befores = []

    def get_alpha(self):
        return (self.x2 - self.x1) / (self.bx2 - self.bx1), (self.y2 - self.y1) / (self.by2 - self.by1)

    def get_theta(self):
        return (self.x1 + self.x2 - self.bx1 - self.bx2) / 2, (self.y1 + self.y2 - self.by1 - self.by2) / 2

    def count_distance(self, other):
        return abs(self.alpha[1] / other.alpha[1]) * abs(self.theta[1] - other.theta[1]) +abs(self.alpha[0] / other.alpha[0]) * abs(self.theta[0] - other.theta[0])

    def add_befores(self,new_before):
        if len(self.befores) < self.N:
            # 如果列表长度小于N，直接添加
            self.befores.append(new_before)
        else:
            # 找到列表中最大的元素
            max_before = max(self.befores, key=lambda x: x.distance)

            # 判断是否需要替换
            if new_before.distance < max_before.distance:
                # 替换最大的元素
                index_of_max = self.befores.index(max_before)
                self.befores[index_of_max] = new_before
def dynamic_program(frames_size, bbxs, ratio, N=10):
    """
    Args
        - frame_size:(width,height)
        - bbxs:[(x1,y1,width,height),...]
        - ratio:(x,y)
        - N:crop nums
    Returns
        - rets:[(crop_regions,score_diffs)...]
    """
    choices = get_choice_from_frame(frames_size, bbxs, ratio)
    nodes = []
    length = len(bbxs)
    for i in range(0, length):
        choice = choices[i]
        lst = []  # lst对应每一帧的可选node
        for prop in choice:
            lst.append(Node(prop, bbxs[i], N))
        nodes.append(lst)

    for i in range(1, length):
        for node_j in nodes[i]:
            for node_k in nodes[i - 1]: # 前一帧的每个node
                if i == 1: # 如果i==1，那么node_k不会有前向节点,distance均为0
                    if len(node_k.befores) == 0:
                        node_k.befores.append(Before(None,node_k,0))
                    node_j.add_befores(Before(node_k.befores[0],node_j,node_j.count_distance(node_k)))
                else: # node_k有若干个前向节点，distance可能不同
                    for before in node_k.befores:
                        node_j.add_befores(Before(before,node_j,node_j.count_distance(node_k)+before.distance))

    # 在最后一帧的若干个node中，找出N个最小的before(可能不到N个)
    find_before_list = []
    for node in nodes[-1]:
        for before in node.befores:
            if len(find_before_list) < N:
                find_before_list.append(before)
            else:
                max_before = max(find_before_list,key=lambda x:x.distance)
                if before.distance < max_before.distance:
                    index_of_max = find_before_list.index(max_before)
                    find_before_list[index_of_max] = before

    # 根据before，保存bbx和scores
    rets = []
    for find_before in find_before_list:
        ret = []
        score = []
        cur = find_before
        cur_node = cur.node
        cur_dis = cur.distance
        while True:
            ret.insert(0, (cur_node.x1, cur_node.y1, cur_node.x2, cur_node.y2))
            if cur.before is None:
                break
            else:
                cur = cur.before
                cur_node = cur.node
                score.insert(0,cur_dis-cur.distance)
                cur_dis = cur.distance

        rets.append((ret,score))
    return rets


def get_choice_from_frame(frames_size, bbxs, ratio):
    """
    根据每帧的bbx，计算出每帧的可选集
    Args
        - frame_size:(width,height)
        - bbxs:[(x1,y1,width,height),...]
        - ratio:(x,y)
    Returns
        - ret:
    """
    length = len(bbxs)
    choices = []
    for i in range(0, length):
        bbx = bbxs[i]
        possible_crops = get_all_possible_crops(frames_size, bbx, ratio)
        choices.append(possible_crops)

    return choices

def get_all_possible_crops(frames_size, crop_region, ratio):
    """
    根据当前帧的bbx，计算出当前帧的可选集
    """
    split_result = split_and_crop_frame(frames_size, crop_region, ratio)
    if split_result is None:
        print("Error:No possible crops")
    ret = []
    for item in split_result:
        ret.append((item[0], item[1], item[2], item[3]))
    return ret


def split_and_crop_frame(frame_size, crop_region, ratio):
    """
    迭代裁切画面，不断移动锚点以适应新的画幅。

    参数：
    - frame_size: 原始图片大小  (width, height)
    - crop_region: 裁切区域，在此项目中为bounding-box
    - ratio: 裁切比例，例如 (1,1)

    返回值：
    返回list (x1,y1,x2,y2)，表示最终可选的裁切集
    """
    # 获取原始图像大小
    original_width, original_height = frame_size

    # 计算裁剪区域的坐标
    x1, y1, width, height = crop_region
    x2 = x1 + width - 1
    y2 = y1 + height - 1

    # 获取裁剪区域的长宽和锚点
    anchor_x = int((x1 + x2) / 2)
    anchor_y = int((y1 + y2) / 2)
    crop_width = (x2 - x1)
    crop_height = (y2 - y1)

    # 调整锚点和大小
    crop_width, crop_height = calculate_final_size((crop_width, crop_height), (ratio[0], ratio[1]),
                                                   set_min_resolution((ratio[0], ratio[1])))
    adjusted_anchor = adjust_anchor((crop_width, crop_height), (anchor_x, anchor_y), (original_width, original_height))

    if adjusted_anchor is None:
        # 说明画面大小不合适
        return None

    # 记录结果
    crop_result = []

    while adjusted_anchor is not None:
        anchor_x, anchor_y = adjusted_anchor
        crop_grid_start_x = anchor_x - int(crop_width / 2)
        crop_grid_end_x = anchor_x + int(crop_width / 2)
        crop_grid_start_y = anchor_y - int(crop_height / 2)
        crop_grid_end_y = anchor_y + int(crop_height / 2)

        crop_result.append([crop_grid_start_x, crop_grid_start_y, crop_grid_end_x, crop_grid_end_y])

        # 扩大并评分
        (crop_width, crop_height), adjusted_anchor = adjust_size_and_anchor((crop_width, crop_height), 1.005,
                                                                            adjusted_anchor,
                                                                            (original_width, original_height))

    return crop_result

def calculate_final_size(original_size, ratio, min_resolution):
    """
    计算裁切画面的最终大小，确保包含原图片。

    参数：
    - original_size: 源图片的大小，格式为 (width, height)
    - ratio: 裁切比例，例如 (1,1)
    - min_resolution: 最小分辨率，格式为 (min_width, min_height)

    返回值：
    返回元组 (final_width, final_height)，表示最终裁切画面的大小。
    """

    # 解析裁切比例

    ratio_x, ratio_y = ratio

    # 解析原始图片大小
    original_width, original_height = original_size

    # 计算按照比例裁切后的宽度和高度
    if original_width * ratio_y > ratio_x * original_height:
        final_width = max(min_resolution[0], original_width)
        final_height = int(final_width / ratio_x * ratio_y)
    else:
        final_height = max(min_resolution[1], original_height)
        final_width = int(final_height / ratio_y * ratio_x)

    return final_width, final_height

def adjust_anchor(final_size, anchor, frame_size):
    """
    调整画面的锚点，确保画面完全包含在更大画面内。

    参数：
    - final_size: 最终裁切画面的大小，格式为 (width, height)
    - anchor: 初始锚点，格式为 (x, y)
    - larger_frame_size: 更大画面的大小，格式为 (width, height)

    返回值：
    返回调整后的锚点 (adjusted_x, adjusted_y)。
    """

    # 解析各个参数
    final_width, final_height = final_size
    anchor_x, anchor_y = anchor
    frame_width, frame_height = frame_size

    # 检查锚点是否超出范围
    if anchor_x - int(final_width / 2) < 0 or anchor_x + int(final_width / 2) > frame_width:
        # 如果锚点超出左侧或右侧，将锚点调整到合适位置
        left_edge = int(final_width / 2)
        right_edge = frame_width - int(final_width / 2)
        if left_edge > right_edge:
            return None
        else:
            if abs(anchor_x - left_edge) < abs(anchor_x - right_edge):
                anchor_x = left_edge
            else:
                anchor_x = right_edge
    if anchor_y - int(final_height / 2) < 0 or anchor_y + int(final_height / 2) > frame_height:
        # 如果锚点超出上侧或下侧，将锚点调整到合适位置
        up_edge = int(final_height / 2)
        down_edge = frame_height - int(final_height / 2)
        if up_edge > down_edge:
            return None
        else:
            if abs(anchor_y - up_edge) < abs(anchor_y - down_edge):
                anchor_y = up_edge
            else:
                anchor_y = down_edge
    return anchor_x, anchor_y

def adjust_size_and_anchor(original_size, ratio, anchor, frame_size):
    """
    使画面变大并调整锚点，确保画面适应在更大画面内。

    参数：
    - original_size: 原始画面的大小，格式为 (width, height)
    - ratio: 变大的比例，例如 1.5 表示增大50%
    - anchor: 锚点，格式为 (x, y)
    - frame_size: 画面的大小，格式为 (width, height)

    返回值：
    返回元组 (adjusted_size, adjusted_anchor)，表示调整后的画面大小和锚点。
    """

    # 计算变大后的画面大小
    new_width = int(original_size[0] * ratio)
    new_height = int(original_size[1] * ratio)

    # 计算新的锚点位置
    anchor_x, anchor_y = anchor

    # 调整锚点和画面大小，确保画面在更大画面内适应指定的锚点
    adjusted_anchor = adjust_anchor((new_width, new_height), (anchor_x, anchor_y), frame_size)

    return (new_width, new_height), adjusted_anchor

def set_min_resolution(resolution):
    """
    根据输入的分辨率设置最小分辨率

    参数：
    - resolution: 分辨率，格式为 (width, height)

    返回值：
    返回元组 (min_width, min_height)，表示对应的最小分辨率。
    """
    if resolution == (1, 1):
        return 120, 120
    elif resolution == (4, 3):
        return 160, 120
    elif resolution == (3, 4):
        return 120, 160
    elif resolution == (16, 9):
        return 192, 108
    elif resolution == (9, 16):
        return 108, 192
    elif resolution == (3, 2):
        return 162, 108
    elif resolution == (2, 3):
        return 108, 162

def read_images(directory_path):
    image_list = []

    # 遍历指定目录下的文件
    for filename in os.listdir(directory_path):
        if filename.endswith(".jpg") and filename[:4].isdigit() and len(filename) == 8:
            # 构建完整的文件路径
            file_path = os.path.join(directory_path, filename)

            # 读取图片并添加到列表
            try:
                img = Image.open(file_path)
                image_list.append(img)
            except Exception as e:
                print(f"Error reading {file_path}: {e}")

    return image_list

def read_txt_file(file_path):
    data_list = []

    try:
        with open(file_path, 'r') as file:
            for line in file:
                # 按空格分割每一行的数据
                data = line.strip().split()

                # 将数据转换为整数
                data = [int(value) for value in data]

                # 添加到数据列表
                data_list.append([data[1],data[2],data[3],data[4]])

    except Exception as e:
        print(f"Error reading {file_path}: {e}")

    return data_list

def original_images_to_video(images, output_video_path, fps=30):
    # 获取第一张图片的大小
    video_width, video_height = images[0].size

    # 设置视频编码器和帧率
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (video_width, video_height))

    for image in images:
        # 将图片转换为BGR格式（OpenCV的视频编码需要BGR格式）
        image_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        video_writer.write(image_bgr)

    video_writer.release()

def images_to_video(images, crop_rectangles, output_video_path,fps=30):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    # Get the size of the first image
    video_width, video_height = images[0].size

    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (video_width, video_height))

    for i in range(len(images)):
        image = np.array(images[i])

        # Check if there are corresponding crop_rectangles
        if i < len(crop_rectangles):
            crop_rectangle = crop_rectangles[i]
            crop_image = image[crop_rectangle[1]:crop_rectangle[3], crop_rectangle[0]:crop_rectangle[2]]
        else:
            # If no crop_rectangle is available, use the entire image
            crop_image = image

        resized_image = cv2.resize(crop_image, (video_width, video_height))
        # Convert to BGR format (OpenCV's video encoding requires BGR format)
        resized_image = cv2.cvtColor(resized_image, cv2.COLOR_RGB2BGR)

        video_writer.write(resized_image)

    video_writer.release()

if __name__=="__main__":
    images = read_images("video_bbxs/ori_images")
    bbxs = read_txt_file("video_bbxs/bounding_boxes.txt")
    frame_size = images[0].size
    rets = dynamic_program(frame_size,bbxs,(3,2))
    crop_region,scores = rets[0]
    print(len(crop_region),len(scores))
    # 修改保存裁剪图像的部分为合并视频
    original_images_path = "crop_images_output/original_video.mp4"
    output_video_path = "crop_images_output/output_video.mp4"
    original_images_to_video(images,original_images_path)
    images_to_video(images,crop_region, output_video_path)