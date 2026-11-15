# Provider 协议

`utils/providers/` 收拢所有供应商专属逻辑。`utils/llm_factory.py` 只负责解析路由并分发，代码里不含任何供应商名称的分支。

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

- `create_chat_model(route, kwargs, *, catalog_profile)`：过滤 SDK 不接受的参数并构造模型。
- `structured_output_method(route, capabilities)`：没有厂商覆盖时的结构化输出策略。

`Vendor`：

- `serves(route)`：路由是否到达本厂商的官方端点。
- `request_overrides(route, *, user_id, cache_key)`：只有官方 API 接受的 `create_llm` 参数。
- `normalize_reasoning_effort(route, effort)`：把通用强度名映射为厂商取值。
- `native_web_search(route)`：返回 `(是否支持, 原因)`；返回 `None` 表示走通用规则。
- `structured_output_method(route, capabilities)`：返回 `None` 表示交给适配器决定。
- `native_search_tools`：启用托管搜索时注入的工具类型。
- `profile_fields`：只有本厂商 API 理解的 profile 字段（值必须是 table），例如 DeepSeek 的 `safety_check` 和 `title`。

`ChatAdapter.default_profiles` 声明该供应商自带的 `[providers.<name>]` 表。配置加载时由 `default_profiles()` 汇总为内置 profile，再合并用户配置；`utils/configs.py` 不再写任何供应商的默认值。

`BaseChatAdapter` 和 `BaseVendor` 提供默认实现，子类只写与默认不同的部分。

## 新增供应商

1. 在 `utils/providers/` 新建模块，继承 `BaseChatAdapter`，声明 `type`、`valid_kwargs`、`api_key_field`、`base_url_field`、`default_profiles` 和 `model_class()`。
2. 在 `base.py` 的 `ADAPTER_API_MODES` 与 `DEFAULT_API_MODES` 登记它支持的协议；配置校验读取同一份表。
3. 在 `utils/providers/__init__.py` 的 `_ADAPTERS` 登记。`_ADAPTERS` 的顺序就是按模型名推断的顺序。
4. 只有需要发送官方专属字段时才实现 `Vendor`，并登记到 `_VENDORS`。`serves()` 必须校验主机名，不能只看适配器类型。
5. 测试替换 SDK 类时 patch 供应商模块，例如 `utils.providers.openai.ChatOpenAI`。

## 约束

- `utils/providers/__init__.py` 与 `base.py` 在导入时不加载任何厂商 SDK；适配器在首次使用时加载（配置加载读取内置 profile 时会触发）。
- `deepseek/media.py` 和 `deepseek/web.py` 读取应用配置，由调用方显式导入，不经包入口再导出。
- 托管媒体接口的 `credentials_profile` 只用于选择密钥；只有 `type = "deepseek"` 的 profile 在 `base_url` 为空时默认使用官方地址，其他 profile 必须显式填写 `base_url`，避免把别家的密钥发到 DeepSeek。
