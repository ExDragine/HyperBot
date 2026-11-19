---
name: daily-news
description: Use when asked to compile a daily news briefing, a roundup of recent headlines, or a scheduled or preview edition of the HyperBot news report.
---

# 每日新闻

合并同一事件，按重要性整理「今日要闻」和「值得一看」。只写证据支撑的事实。日报用简体中文，不为政治选择排名，不预测选举结果。

## 按任务模式执行

- `mode=research` 时，按 `window` 和主题使用可用的 DeepSeek 服务端搜索。核对发布日期和事件发生时间，打开重点报道。证据的 `article_id` 填原始来源 URL，`quote` 逐字取自服务端搜索摘要。网页摘要只用于辅助核对。证据不足时只保留有合格证据的条目。
- `mode=revision` 时，只使用给定的 `articles` 修订，不进行联网检索。原样使用资料中的 `article_id`，保留逐字引文，按 `validation_error` 修正错误。
- 普通聊天按请求中的时间窗口和主题用本轮可用工具查证，按北京时间理解「今天」。材料不足时减少条目并说明。用户要求订阅或定时推送时，按定时任务权限处理。

归档和预览以 `target_stories` 为条数目标，两个栏目分别最多 6 条和 12 条。同一来源只支持一条新闻，补充证据也计入。归档和群投递由调用方处理。
