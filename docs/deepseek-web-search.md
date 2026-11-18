# DeepSeek 服务端搜索与网页读取

官方 DeepSeek Flash Responses 路由会在主模型的请求中声明 `deepseek_web_search` 和 `deepseek_read_url`。DeepSeek 在服务端决定何时搜索及读取网页，并返回来源、引用和网页读取记录。费用按模型 Token 用量计算。 [DeepSeek 联网工具](https://api-docs.deepseek.com/zh-cn/guides/web_tools/) [官方价格](https://api-docs.deepseek.com/zh-cn/quick_start/pricing/)

## 配置

```toml
[models]
advanced_model = "deepseek-flash"
advanced_model_provider = "deepseek_responses"

[providers.deepseek_responses]
type = "openai"
api_mode = "responses"
base_url = "https://api.deepseek.com"
api_key = ""
```

API Key 配在模型使用的 profile 中。`native_web_search = false` 可关闭该 profile 的服务端工具。旧 `[search]` 配置仍可读取，当前联网能力由主模型路由决定。

## 每日新闻

聊天中的日报请求由主 Agent 按需读取 `daily-news` 技能。定时和 `/news preview` 使用同一技能，并校验实际返回的搜索摘要、来源 URL 与日期。网页读取返回的模型总结用于核对，逐字引文由搜索摘要验证。 [每日新闻技能](../skills/daily-news/SKILL.md) [新闻插件运维](news-plugin.md)

## 验证

真实 SDK 契约测试使用模拟 HTTP 响应验证本地函数与 DeepSeek 服务端工具共存，并检查服务端证据字段。

```powershell
uv run --locked pytest test/integration/deepseek_search_runtime_test.py test/news -q
```
