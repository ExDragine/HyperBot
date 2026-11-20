---
name: satellite-imagery
description: 用于查询地球气象卫星云图，包括风云四号 FY-4B 和葵花卫星 Himawari。
metadata:
  include_tools: get_fy4b_satellite_image get_fy4b_cloud_map get_fy4b_geos_cloud_map get_himawari_satellite_image
---

# 气象卫星云图

- 风云四号按产品选择 `get_fy4b_satellite_image`、`get_fy4b_cloud_map` 或 `get_fy4b_geos_cloud_map`，使用用户给出的区域和时次。
- 葵花卫星整图用 `get_himawari_satellite_image`。
- 用工具返回的观测时间标注云图。区域或时间无可用图像时，说明数据缺口。
