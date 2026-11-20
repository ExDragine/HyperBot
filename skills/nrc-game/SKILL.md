---
name: nrc-game
description: 用于洛克王国 NRC 的活动日历、当前商人、精灵蛋组、孵蛋搭配和孵化详情。
metadata:
  include_tools: get_nrc_event_calendar get_nrc_merchant_current get_nrc_eggs_groups get_nrc_eggs_details
---

# 洛克王国 NRC

- 活动排期用 `get_nrc_event_calendar`，当前商人和商品用 `get_nrc_merchant_current`。
- 比较精灵能否孵蛋或查找配偶用 `get_nrc_eggs_groups`，传入正确的精灵名称。
- 查询具体配对的孵化详情用 `get_nrc_eggs_details`，区分父母精灵。
- 查询结果有图片时让系统投递工件。以工具返回的时间和可用数据为准。
