import os
import scipy.io
import csv
import numpy as np

def process_mat_files(input_folder, output_csv):
    with open(output_csv, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow(['filename', 'count', 'locations'])

        for filename in sorted(os.listdir(input_folder)):  # 按文件名排序
           if filename.endswith('.mat'):
                # 生成第一列：去除.mat和GT_前缀，添加.jpg
                base_name = filename.replace('.mat', '').replace('GT_', '') + '.jpg'
                
                try:
                    mat_data = scipy.io.loadmat(os.path.join(input_folder, filename))
                    image_info = mat_data['image_info'][0,0]
                    number = int(image_info['number'][0,0])  # 第二列：纯数字
                    locations = image_info['location']
                    
                    # 处理坐标数据
                    if locations.shape == (1, 1) and isinstance(locations[0,0], np.ndarray):
                        actual_locations = locations[0,0]
                        coord_list = [(float(loc[1]), float(loc[0])) for loc in actual_locations]
                    elif locations.ndim == 2 and locations.shape[1] == 2:
                        coord_list = [(float(loc[1]), float(loc[0])) for loc in locations]
                    else:
                        coord_list = []
                    
                    # 格式化坐标字符串
                    coord_str = "[" + ", ".join([f"({x:.2f}, {y:.2f})" for x, y in coord_list]) + "]"
                    
                    # 写入行，确保无多余空格
                    csv_writer.writerow([base_name, str(number), coord_str])
                
                except Exception as e:
                    print(f"处理文件 {filename} 时出错: {str(e)}")
                    continue
    
    print(f"处理完成，结果已保存到 {output_csv}")

# 使用示例
input_folder = 'ground_truth'  # 替换为你的.mat文件所在文件夹
output_csv = '1gt.csv'         # 输出CSV文件名
process_mat_files(input_folder, output_csv)