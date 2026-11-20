---
name: space-weather
description: 用于极光、太阳活动、太阳风、地磁空间环境、太阳影像和行星天气。
metadata:
  include_tools: aurora_live solar_flare realtime_solarwind soho_realtime_solarwind geospace noaa_enlil_predict solar_image goes_suvi sunspot swpc_page planets_weather mars_weather
---

# 空间天气

- 极光观测用 `aurora_live`。耀斑用 `solar_flare`。太阳风优先用 `realtime_solarwind`，需要 SOHO 数据时用 `soho_realtime_solarwind`。
- 地磁空间环境用 `geospace`，太阳风传播预报用 `noaa_enlil_predict`，综合监测页用 `swpc_page`。
- 按需求选择 `solar_image`、`goes_suvi` 或 `sunspot`，说明图像来源和时间。
- 火星地面天气用 `mars_weather`。其他行星的观测先看 `planets_weather` 支持的对象。
- 区分观测和预报，缺少观测时间或所问区域的数据时直接说明。
