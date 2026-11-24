# DeepSeek 服务端搜索与网页读取

官方 DeepSeek Flash 的 Chat Completions、Responses 和 Anthropic Messages 路由都会声明 `deepseek_web_search` 和 `deepseek_read_url`。DeepSeek 在服务端决定何时搜索及读取网页，并返回来源、引用和网页读取记录。费用按模型 Token 用量计算。思考模式下，同时声明两个工具时，DeepSeek 使用深度思考搜索，其中可以打开搜索结果。 [DeepSeek 联网工具](https://api-docs.deepseek.com/zh-cn/guides/web_tools/) [官方价格](https://api-docs.deepseek.com/zh-cn/quick_start/pricing/)

## 配置

```toml
[models]
advanced_model = "deepseek-flash"
advanced_model_provider = "deepseek"

[providers.deepseek]
type = "deepseek"
api_mode = "chat_completions"
base_url = "https://api.deepseek.com"
api_key = ""
```

使用 Responses 时将 `advanced_model_provider` 设为 `deepseek_responses`，使用 Anthropic Messages 时设为 `deepseek_anthropic`。在选中的 profile 配置 API Key。`native_web_search = false` 可关闭该 profile 的服务端工具。

独立工具 `deepseek_web_search` 和 `deepseek_web_fetch` 由 HyperBot 调用 DeepSeek Search / Fetch 接口，复用 `[search].provider` 的 API Key。默认复用 `deepseek_responses`，已有 `[search]` 配置可以继续使用。`[search].enabled = false` 可关闭独立工具。

DeepSeek 服务端联网失败并触发回退时，当前轮会改用独立工具。主模型无服务端联网时，已配置的独立工具可以直接使用。搜索默认返回 20 条，网页正文过长时可通过 `content_path` 读取全文。独立接口按成功请求计费。 [DeepSeek Search & Fetch](https://api-docs.deepseek.com/zh-cn/guides/search/)

## 每日新闻

聊天中的日报请求由主 Agent 按需读取 `daily-news` 技能。定时和 `/news preview` 使用同一技能，并校验实际返回的搜索摘要、来源 URL 与日期。网页读取返回的模型总结用于核对，逐字引文由搜索摘要验证。 [每日新闻技能](../skills/daily-news/SKILL.md) [新闻插件运维](news-plugin.md)

## 验证

真实 SDK 契约测试通过模拟 HTTP 验证三种协议都能同时使用本地函数和服务端联网，并覆盖来源证据、引用和同轮回退。

```powershell
uv run --locked pytest test/integration/deepseek_search_runtime_test.py test/integration/deepseek_messages_runtime_test.py test/news -q
```
