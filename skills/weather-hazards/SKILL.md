---
name: weather-hazards
description: 用于查询中国天气雷达、台风路径、风切变和涡旋图，以及国内外地震信息。
metadata:
  include_tools: get_available_china_radar_areas get_static_china_radar get_typhoon_info get_wind_map get_china_earthquake get_usgs_significant_earthquakes
---

# 地面气象与地震

- 中国天气雷达用 `get_static_china_radar`。不确定地区名称时先用 `get_available_china_radar_areas`。
- 台风动态和路径用 `get_typhoon_info`，按用户指定的名称和叠加图层查询。只查风切变或涡旋图时用 `get_wind_map`。
- 国内地震用 `get_china_earthquake`，全球显著地震用 `get_usgs_significant_earthquakes`，说明数据时间和来源。
- 区分实时观测、预报和历史记录。雷达和云图只说明图上显示的信息。
