# Provider 协议

`utils/providers/` 按供应商组织实现。`utils/llm_factory.py` 解析路由并通过注册表分发，主 Agent 使用对应路由提供的能力。

## 两个角色

| 角色 | 负责什么 | 以什么为键 | 现有实现 |
| --- | --- | --- | --- |
| `ChatAdapter` | 构造 LangChain 模型：SDK 类、可接受的构造参数、参数改名、默认的结构化输出策略 | `[providers.<name>].type` | `openai`、`google`、`anthropic`、`deepseek` |
| `Vendor` | 官方服务的请求语义：识别官方端点、专属请求字段、推理强度取值、托管搜索、结构化输出覆盖 | 官方端点 | `deepseek`、`openai`、`anthropic` |

两者分开，是因为它们各自独立变化：DeepSeek 一家厂商可经三种适配器到达（`deepseek`、`openai` Responses、`anthropic` Messages）；`openai` 适配器又同时服务任意兼容代理，这些代理不能收到 OpenAI 专属字段。没有 `Vendor` 认领的路由一律按兼容代理处理，不发送任何专属字段。

## Route

工厂把 profile 与协议解析一次，得到 `Route(model, profile, adapter, api_mode, model_family)`。适配器和厂商只接收 `Route`，不再各自从原始 profile 推导 `type` 或 `api_mode`。

## 协议方法

`ChatAdapter`：

- `create_chat_model(route, kwargs, *, catalog_profile, vendor)` 过滤 SDK 参数、应用已匹配厂商的扩展并构造模型。
- `structured_output_method(route, capabilities)` 选择适配器的默认结构化输出策略。
- `structured_output_options(method)` 将策略映射到 SDK 参数。OpenAI 适配器在 JSON Schema 模式下启用 `strict`。
- `api_modes` 和 `default_api_mode` 声明支持的协议和缺省协议，路由和配置校验读取同一份声明。

`Vendor`：

- `serves(route)`：路由是否到达本厂商的官方端点。
- `request_overrides(route, *, user_id, cache_key)`：只有官方 API 接受的 `create_llm` 参数。
- `normalize_reasoning_effort(route, effort)`：把通用强度名映射为厂商取值。
- `native_web_search(route)`：返回 `(是否支持, 原因)`；返回 `None` 表示走通用规则。
- `structured_output_method(route, capabilities)`：返回 `None` 表示交给适配器决定。
- `native_search_tools` 声明启用托管搜索后可注入的工具类型。
- `prepare_chat_model(route, model_class, params)` 让官方端点按需扩展适配器。DeepSeek 在 Flash 的三种协议上用它保留原生联网工具的请求类型。Chat Completions 适配器也负责思考内容回传与工具图片的传输，Messages 适配器负责保留读链签名等原始内容块字段。
- `web_search_middleware(...)` 可为官方路由提供专属联网中间件。DeepSeek 在这里处理同名工具、短追问和同一轮内的降级。
- `standalone_web_search_hint(tool_names)` 根据实际暴露的独立工具提供提示。
- `decision_backend(route, ...)` 提供已启用的官方原生决策接口。OpenAI Decisions 的 SDK 调用和负载转换位于 `openai/decisions.py`。
- `profile_fields`：只有本厂商 API 理解的 profile 字段（值必须是 table），例如 DeepSeek 的 `safety_check` 和 `title`。

`ChatAdapter.default_profiles` 声明该供应商自带的 `[providers.<name>]` 表。配置加载时由 `default_profiles()` 汇总内置 profile，再合并用户配置。

`BaseChatAdapter` 和 `BaseVendor` 提供默认实现，子类只写与默认不同的部分。

## 新增供应商

1. 在 `utils/providers/` 新建供应商包，实现 `BaseChatAdapter` 并声明其协议、内置 profile 和 SDK 参数。
2. 在 `utils/providers/__init__.py` 的 `_ADAPTERS` 登记。`_ADAPTERS` 的顺序就是按模型名推断的顺序。
3. 厂商提供官方端点扩展时实现 `Vendor`，并登记到 `_VENDORS`。`serves()` 根据协议和实际主机名识别官方端点。
4. 测试替换 SDK 类时 patch 供应商模块，例如 `utils.providers.openai.ChatOpenAI`。

## 约束

- `utils/providers/__init__.py` 与 `base.py` 在导入时不加载任何厂商 SDK；适配器在首次使用时加载（配置加载读取内置 profile 时会触发）。
- DeepSeek 的余额和手动 OCR 工具放在 `deepseek/tools/`，由 Provider 注册表加入主 Agent 的直接工具。
- DeepSeek 的余额、OCR、语音及独立联网接口由各服务模块实现。`openai/images.py` 接入官方 Images API 和兼容网关绘图。`openai/videos.py` 只接入兼容视频网关。共享绘图和视频服务仍是应用入口。
- `openai/audio.py` 提供文件转写与 Realtime 合成，`openai/platform.py` 提供显式调用的会话、后台任务、批处理、嵌入和内容审核。见 [OpenAI 媒体和平台接口](openai-platform.md)。
- DeepSeek 托管媒体接口的 `credentials_profile` 只用于选择密钥；只有 `type = "deepseek"` 的 profile 在 `base_url` 为空时默认使用官方地址，其他 profile 必须显式填写 `base_url`，避免把别家的密钥发到 DeepSeek。
