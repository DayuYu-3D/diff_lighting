import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay

# 生成随机的二维点云
np.random.seed(0)
points = np.random.rand(100, 2)

# 进行三角剖分
triangulation = Delaunay(points)

# 获取三角形的顶点索引
triangles = triangulation.simplices

# 可视化结果
plt.triplot(points[:, 0], points[:, 1], triangles)
plt.plot(points[:, 0], points[:, 1], 'o')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Delaunay Triangulation')
# 保存结果为图像文件
plt.savefig('123.png')

# 不显示图形
plt.close()
