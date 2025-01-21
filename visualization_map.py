import os
import cv2
import numpy as np
from utils.util import show_fill_map
# 图片文件夹路径
# folder_path = '/data3/nmy/mirror_pad_256_simplfy/fixed_size_npy'

# # 遍历文件夹中的图片文件
# for filename in os.listdir(folder_path):
#     if filename.endswith(('.jpg', '.jpeg', '.png','.npy')):
#         # 读取图片
#         image_path = os.path.join(folder_path, filename)
#         # image = cv2.imread(image_path,0)
#         map = np.load(os.path.join(folder_path, filename),allow_pickle=True)
           
#         tmp = show_fill_map(map)
#         output_path = os.path.join(folder_path, 'visualize_' + filename)
#         cv2.imwrite(output_path, tmp)
#         # 调整大小为256x256像素
#         # image_resized = cv2.resize(image, (256, 256)).astype('uint8')

#         # # 将线稿变为黑色
#         # gray = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY)
#         # _, threshold = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
#         # image_resized[threshold == 0] = [0, 0, 0]  # 设置线稿为黑色

#         # 保存处理后的图片
        
#         #cv2.imwrite(output_path, image_resized)

#         print(f"Processed image: {filename}")

# print("Image resizing and line drawing completed.")


import os
import numpy as np
import cv2

# 输入文件夹和输出文件夹路径
folder_path = '/data3/nmy/opensource/test_map'

output_folder = '/data3/nmy/opensource/test_map'

# 创建输出文件夹
os.makedirs(output_folder, exist_ok=True)

# 遍历输入文件夹中的所有文件
for filename in os.listdir(folder_path):
    if filename.endswith('.npy'):
        # 构建输入文件路径
        input_path = os.path.join(folder_path, filename)

        # 从.npy文件加载Numpy数组
        #array = np.load(input_path)


        image_path = os.path.join(folder_path, filename)
        # image = cv2.imread(image_path,0)
        map = np.load(os.path.join(folder_path, filename),allow_pickle=True)
        print(map.size)
        tmp = show_fill_map(map)
        # output_path = os.path.join(folder_path, 'visualize_' + filename)
        # cv2.imwrite(output_path, tmp)
        # # 将数组转换为图像（假设数组值范围在0到255之间）
        # image = array.astype(np.uint8)

        # 构建输出文件路径
        output_filename = os.path.splitext(filename)[0] + '.png'
        output_path = os.path.join(output_folder, output_filename)

        # 保存图像为.png文件
        cv2.imwrite(output_path, tmp)