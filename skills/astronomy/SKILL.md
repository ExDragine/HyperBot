---
name: astronomy
description: 用于观测彗星、查询近期火箭发射计划或查看国际空间站和天宫的位置。
metadata:
  include_tools: comet_information comet_list get_launches station_location
---

# 天文观测与发射

- 已知彗星名称时用 `comet_information`。找适合观测的彗星时用 `comet_list`，说明筛选采用的星等上限。
- 查询未来发射用 `get_launches`，按用户给的时间范围填写天数，注明计划时间可能调整。
- 查看空间站位置用 `station_location`，接受「国际空间站」或「天宫」，使用返回的位置图。
