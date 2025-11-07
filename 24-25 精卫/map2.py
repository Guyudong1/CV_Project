import numpy as np
import cv2

# ----------导入图片------------
img_path = "test/test8.jpg"
img = cv2.imread(img_path)

if img is None:
    print(f"错误：无法读取图片 '{img_path}'")
    exit()

fixed_size = (450, 300)
img = cv2.resize(img, fixed_size)
print(f"图片加载成功，尺寸: {img.shape}")

# ----------预处理------------
blurred = cv2.GaussianBlur(img, (1, 1), 0)

# ----------边缘检测------------
gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 50, 150)

# ----------边缘增强和连接------------
kernel = np.ones((3, 3), np.uint8)
edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
edges = cv2.dilate(edges, kernel, iterations=2)

# ----------轮廓检测------------
contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# 设置最小轮廓面积阈值
min_contour_area = 10
filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_contour_area]

print(f"检测到轮廓数量: {len(filtered_contours)}")

# ----------找到最大的轮廓------------
if filtered_contours:
    # 按面积排序，找到最大的轮廓
    largest_contour = max(filtered_contours, key=cv2.contourArea)
    largest_area = cv2.contourArea(largest_contour)
    print(f"最大轮廓面积: {largest_area}")

    # 只保留最大的轮廓作为岛屿
    island_contours = [largest_contour]
    ocean_contours = []  # 不显示其他轮廓的边界
else:
    island_contours = []
    ocean_contours = []
    print("未检测到轮廓")

# ----------创建分割结果------------
img_boundary = img.copy()
# 只绘制最大的轮廓边界
cv2.drawContours(img_boundary, island_contours, -1, (0, 255, 0), 2)  # 绿色-岛屿边界

# 创建分割掩码 - 只填充最大的轮廓
island_mask = np.zeros(img.shape[:2], np.uint8)
cv2.drawContours(island_mask, island_contours, -1, 255, thickness=cv2.FILLED)

# 海域掩码是岛屿掩码的反转
ocean_mask = cv2.bitwise_not(island_mask)

# ----------创建简易地图------------
img_map = np.zeros(img.shape, np.uint8) + 255  # 白色背景

# 填充岛屿区域（浅绿色）
img_map[island_mask == 255] = (144, 238, 144)  # 浅绿色-岛屿

# 填充海域区域（浅蓝色）
img_map[ocean_mask == 255] = (255, 200, 150)  # 浅蓝色-海域

# 绘制最大的轮廓边界
cv2.drawContours(img_map, island_contours, -1, (0, 100, 0), 2)  # 深绿色岛屿边界

# ----------地图栅格化----------
img_map2 = img_map.copy()
height, width = img_map2.shape[:2]
grid_size = 20
color_gray = (128, 128, 128)

for x in range(0, width, grid_size):
    cv2.line(img_map2, (x, 0), (x, height), color_gray, 1)
for y in range(0, height, grid_size):
    cv2.line(img_map2, (0, y), (width, y), color_gray, 1)

# ----------结果显示------------
print(f"显示最大轮廓数量: {len(island_contours)}")

# 显示结果
cv2.imshow("Original", img)
cv2.imshow("Canny Edges", edges)

# 创建对比图：原图 | 边界图 | 填充图
res = np.hstack((img, img_boundary, img_map))
cv2.imshow("Comparison: Original | Boundary | Filled Map", res)
cv2.imshow("Final Map", img_map)
cv2.imshow("Grid Map", img_map2)

print("按任意键关闭窗口...")
cv2.waitKey(0)
cv2.destroyAllWindows()