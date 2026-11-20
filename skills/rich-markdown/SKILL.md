---
name: rich-markdown
description: Compose longer QQ replies, comparisons, guides and data or media with Markdown and static components.
---

# 先判断是否需要排版

简单聊天和需要原样复制的内容直接写正文。需要解释、比较、组织数据或展示媒体时，再使用 Markdown 和下方组件。普通文本默认达到 500 字符时由系统转图，回复篇幅由任务需要决定。

## 基本组件语法

组件打开行为 `::名称{属性}`，结束行为独立的 `::`，正文直接写 Markdown。文字属性加引号，数字直接写；无属性时省略花括号。

```markdown
::card{title="核心判断" color="blue"}
这里写 **重点** 和自然段落，不需要填写 type/children/text。
::
```

`card` 默认柔和底色，可选 color：neutral/blue/emerald/violet/amber/rose/cyan；variant：muted/outline/ghost。`section` 是无边框分区，可填 title；`grid{columns=2}` 放两列短内容，窄图自动纵向排列。组件可嵌套，每个打开行都要有对应关闭行。基本卡片和分区按上述写法直接输出，无需先读取额外文件。

## 按内容关系选形式

- **解释概念或原因**：沿因果写连贯正文，按机制与应用分区；关键判断可用单张卡片强调，关系确实需要图时加 Mermaid。
- **比较对象**：少量对象的短概览可用一个 grid 组合卡片。共同维度多或对象多时，用 Markdown 表格对齐比较项，选择依据写在正文。
- **给出方法**：使用有序列表，步骤较完整时用 `::steps`；每步包含操作、预期结果或检查方法，命令保留 Markdown 代码块。
- **属性和引文**：需要展示属性或原文引语时，用普通 Markdown 列表或引用。
- **数据与事件**：确有依据才使用 chart/stats/timeline 的 JSON 代码块，可以放在分区内部。没有数值时用正文或表格，不编造比例、指标或进度。
- **位置与素材**：使用独立整行的 map/image/iframe/three/flow；地图只填写查证过的 WGS84 坐标，Turf.js 分析、地图图层、三维对象和流程图使用声明式字段，最终网页仍输出静态快照。`three` 与 `flow` 必须使用对应语言标记的 JSON fenced block，不要写成普通 `json` fenced block。

不要为所有回答规定同一套模块或顺序。段落承担完整论证，卡片承载独立对象或重点，不包住每段文字。让标题说明信息关系，避免空泛的“概述／详情／总结”。长说明、宽表格、步骤和媒体纵向占整行，两列只放短内容。颜色用于区分主题，不暗示不存在的好坏。

## 按需阅读契约

以下路径相对 `/skills/rich-markdown/`：

- 复杂嵌套、完整属性或 `steps` 写法：读取 [组件契约](references/components.md)。
- chart/stats/timeline：读取 [数据契约](references/data.md)。
- map/image/iframe/three/flow：读取 [媒体契约](references/media.md)。
- 组合拿不准时选读 [解释](examples/article.md)、[比较](examples/comparison.md) 或 [指南](examples/guide.md)，只参考结构，不复制事实或固定顺序。
- 只有明确要求旧格式、需要旧格式独有组件或必须兼容旧调用方时，才按 [旧契约](references/legacy-ui.md) 输出 `ui` JSON。

只使用列出的组件与字面量属性。原始 HTML、CSS、JavaScript、表达式、事件和交互控件不受支持。Leaflet 和 Three.js 使用受控声明式字段。检查正文完整、组件配对、颜色含义与来源。
