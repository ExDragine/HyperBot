---
name: weather-hazards
description: 用于查询中国天气雷达、台风路径、风切变和涡旋图、风云四号和葵花卫星云图，以及国内外地震信息。
metadata:
  include_tools: get_available_china_radar_areas get_static_china_radar get_typhoon_info get_wind_map get_china_earthquake get_usgs_significant_earthquakes get_fy4b_satellite_image get_fy4b_cloud_map get_fy4b_geos_cloud_map get_himawari_satellite_image
---

# 气象观测与地震

## 雷达、台风和风场

- 中国天气雷达用 `get_static_china_radar`。不确定地区名称时先用 `get_available_china_radar_areas`。
- 台风动态和路径用 `get_typhoon_info`，按用户指定的名称和叠加图层查询。只查风切变或涡旋图时用 `get_wind_map`。

## 气象卫星云图

- 风云四号按产品选择 `get_fy4b_satellite_image`、`get_fy4b_cloud_map` 或 `get_fy4b_geos_cloud_map`，使用用户给出的区域和时次。
- 葵花卫星整图用 `get_himawari_satellite_image`。
- 用工具返回的观测时间标注云图。区域或时间无可用图像时，说明数据缺口。

## 地震

- 国内地震用 `get_china_earthquake`，全球显著地震用 `get_usgs_significant_earthquakes`，说明数据时间和来源。

区分实时观测、预报和历史记录。雷达和云图只说明图上显示的信息。
