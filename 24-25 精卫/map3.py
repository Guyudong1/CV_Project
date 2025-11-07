import numpy as np
import cv2
import os

def load_and_preprocess_image(img_path):
    """导入并预处理图片"""
    img = cv2.imread(img_path)
    if img is None:
        print(f"错误：无法读取图片 '{img_path}'")
        return None

    fixed_size = (450, 300)
    img = cv2.resize(img, fixed_size)
    print(f"图片加载成功，尺寸: {img.shape}")
    return img


def color_segmentation_method(img):
    """方法1：颜色分割"""
    img_1 = img.copy()

    # -----------K-means 调整对比度-----------
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    lab = cv2.cvtColor(img_1, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l_enhanced = clahe.apply(l)
    lab_enhanced = cv2.merge((l_enhanced, a, b))
    img2 = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)

    # ----------将RGB转变BGR-----------
    img2 = cv2.cvtColor(img_1, cv2.COLOR_RGB2BGR)

    # ----------转换HSV颜色空间-----------
    hsv = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)

    # ----------设定颜色范围------------
    lower_red1 = np.array([0, 100, 45])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 100, 50])
    upper_red2 = np.array([180, 255, 255])
    lower_orange = np.array([10, 50, 10])
    upper_orange = np.array([25, 255, 255])
    lower_yellow = np.array([20, 100, 110])
    upper_yellow = np.array([40, 255, 255])
    lower_brown = np.array([10, 50, 50])
    upper_brown = np.array([30, 255, 200])

    # ----------创建暖色调掩码------------
    mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask_orange = cv2.inRange(hsv, lower_orange, upper_orange)
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
    mask_brown = cv2.inRange(hsv, lower_brown, upper_brown)

    warm_mask = mask_red1 | mask_red2 | mask_orange | mask_yellow | mask_brown
    island_mask = cv2.bitwise_not(warm_mask)

    # ----------图像处理------------
    kernel = np.ones((9, 9), np.uint8)
    island_mask_processed = cv2.morphologyEx(island_mask, cv2.MORPH_CLOSE, kernel)

    return island_mask_processed


def edge_detection_method(img):
    """方法2：边缘检测"""
    img_edge = img.copy()
    blurred = cv2.GaussianBlur(img_edge, (3, 3), 0)
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)

    # 边缘增强
    kernel_edge = np.ones((3, 3), np.uint8)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel_edge)
    edges = cv2.dilate(edges, kernel_edge, iterations=1)

    # 从边缘创建轮廓
    contours_edge, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    island_mask = np.zeros(img.shape[:2], np.uint8)

    # 过滤小轮廓
    min_area = 50
    for cnt in contours_edge:
        if cv2.contourArea(cnt) > min_area:
            cv2.drawContours(island_mask, [cnt], -1, 255, thickness=cv2.FILLED)

    return island_mask


def fuse_methods(mask_color, mask_edge, weight_color=0.8, weight_edge=0.2, threshold=0.4):
    """融合两种方法：加权结合"""
    # 转换为浮点数并加权
    mask_color_float = mask_color.astype(np.float32) / 255.0 * weight_color
    mask_edge_float = mask_edge.astype(np.float32) / 255.0 * weight_edge

    # 加权结合
    combined_float = mask_color_float + mask_edge_float

    # 阈值处理
    combined_mask = (combined_float > threshold).astype(np.uint8) * 255

    # 后处理
    kernel = np.ones((7, 7), np.uint8)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
    # combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)

    # 找到最大轮廓
    contours_final, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours_final:
        largest_contour = max(contours_final, key=cv2.contourArea)
        final_mask = np.zeros(combined_mask.shape[:2], np.uint8)
        cv2.drawContours(final_mask, [largest_contour], -1, 255, thickness=cv2.FILLED)
        return final_mask
    else:
        return combined_mask


def create_final_map(img, island_mask):
    """创建最终的地图"""
    # 创建边界图像
    img_boundary = img.copy()
    contours_boundary, _ = cv2.findContours(island_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img_boundary, contours_boundary, -1, (0, 255, 0), 2)

    # 创建海域掩码
    ocean_mask = cv2.bitwise_not(island_mask)

    # 创建地图
    img_map = np.zeros(img.shape, np.uint8) + 255
    img_map[island_mask == 255] = (144, 238, 144)  # 岛屿-绿色
    img_map[ocean_mask == 255] = (255, 200, 150)  # 海域-蓝色
    cv2.drawContours(img_map, contours_boundary, -1, (0, 100, 0), 2)

    return img_boundary, img_map


def combined_segmentation(img_path):
    """主函数：结合两种方法进行分割"""
    # 1. 导入图片
    img = load_and_preprocess_image(img_path)
    if img is None:
        return None
    # 2. 方法1：颜色分割
    mask_color = color_segmentation_method(img)
    # 3. 方法2：边缘检测
    mask_edge = edge_detection_method(img)
    # 4. 融合两种方法
    final_mask = fuse_methods(mask_color, mask_edge)
    # 5. 创建最终地图
    img_boundary, img_map = create_final_map(img, final_mask)
    # 显示中间结果
    res1 = np.hstack((mask_color, mask_edge, final_mask))
    cv2.imshow("Color Mask|Edge Mask|Combined Mask", res1)
    # 显示最终结果
    res2 = np.hstack((img, img_boundary, img_map))
    cv2.imshow("Combined: Original | Boundary | Map", res2)
    # 统计信息
    contours_boundary, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_edge, _ = cv2.findContours(mask_edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    print(f"颜色分割轮廓数: {len(contours_edge)}")
    print(f"最终轮廓数: {len(contours_boundary)}")

    return res1, res2, img_map


def save_segmentation_results(result1, result2, img_path, save_dir):
    """保存分割结果到指定文件夹"""
    # 获取图片文件名（不含扩展名）
    img_name = os.path.splitext(os.path.basename(img_path))[0]

    # 确保保存文件夹存在
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 保存两张结果图 - 这里就是 {图片名}_res_1 和 {图片名}_res_2
    res1_path = os.path.join(save_dir, f"{img_name}_res_1.jpg")
    res2_path = os.path.join(save_dir, f"{img_name}_res_2.jpg")

    cv2.imwrite(res1_path, result1)  # 掩码对比图
    cv2.imwrite(res2_path, result2)  # 最终结果图

    print(f"结果1已保存: {res1_path}")
    print(f"结果2已保存: {res2_path}")

    return res1_path, res2_path


def process_and_save_all_images(input_folder, output_folder):
    """处理文件夹中的所有图片并保存结果"""
    # 获取所有支持的图片文件
    supported_formats = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif')
    image_files = []

    for file in os.listdir(input_folder):
        if file.lower().endswith(supported_formats):
            image_files.append(file)

    if not image_files:
        print(f"在文件夹 '{input_folder}' 中没有找到图片文件")
        return

    print(f"找到 {len(image_files)} 张图片需要处理:")
    for img_file in image_files:
        print(f"  - {img_file}")

    # 处理每张图片
    for i, img_file in enumerate(image_files, 1):
        img_path = os.path.join(input_folder, img_file)
        print(f"\n[{i}/{len(image_files)}] 正在处理: {img_file}")

        try:
            # 处理图片
            result1, result2, _ = combined_segmentation(img_path)

            if result1 is not None and result2 is not None:
                # 保存结果
                save_segmentation_results(result1, result2, img_path, output_folder)
                print(f"✓ 完成处理: {img_file}")
            else:
                print(f"✗ 处理失败: {img_file} (返回了None)")

        except Exception as e:
            print(f"✗ 处理失败: {img_file}, 错误: {e}")
            continue

    print(f"\n所有图片处理完成！结果保存在 '{output_folder}' 文件夹中")


def save_all_maps(input_folder="img", output_folder="img"):
    """批量保存所有图片的地图"""
    # 获取所有图片文件
    image_files = [f for f in os.listdir(input_folder)
                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    if not image_files:
        print("没有找到图片文件")
        return

    print(f"开始处理 {len(image_files)} 张图片的地图...")

    for img_file in image_files:
        img_path = os.path.join(input_folder, img_file)
        print(f"处理: {img_file}")

        try:
            # 处理图片获取地图
            _, _, img_map = combined_segmentation(img_path)

            if img_map is not None:
                # 保存地图
                img_name = os.path.splitext(img_file)[0]
                map_path = os.path.join(output_folder, f"{img_name}_map.jpg")
                cv2.imwrite(map_path, img_map)
                print(f"✓ 保存: {map_path}")
            else:
                print(f"✗ 失败: {img_file} (无地图输出)")

        except Exception as e:
            print(f"✗ 错误: {img_file} - {e}")

    print("所有地图保存完成！")


if __name__ == "__main__":
    # 使用保存函数处理所有图片
    # input_folder = "res"
    # output_folder = "res"
    # process_and_save_all_images(input_folder, output_folder)

    input_folder = "img"
    output_folder = "img"
    save_all_maps(input_folder, output_folder)
