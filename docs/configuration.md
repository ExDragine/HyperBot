# 配置参考

HyperBot 的应用配置在 `env.toml`。从 [完整配置示例](../env.toml.example) 复制后，填写主模型、辅助模型和所用供应商的密钥，并设置 Dashboard 密码。其余字段按需修改，省略的段和字段使用默认值。真实配置文件含密钥，不进入版本库。

本页描述当前实现；字段校验以 `utils/configs.py` 为准，能力与路由以 `utils/llm_factory.py` 和各服务代码为准。

## 配置文件

| 文件 | 职责 | 生效方式 |
| --- | --- | --- |
| `.env` | NoneBot 环境变量：`HOST`、`PORT`、`DRIVER`、`SUPERUSERS`、`NICKNAME` 等 | 进程启动时读取，修改后重启 |
| `env.toml` | HyperBot 应用配置：人设、模型、供应商、密钥、功能开关、策略、限额、存储和 Dashboard | 启动时校验；Dashboard 保存后原子重载，后续请求/任务读取新值；手工编辑需重启 |
| `mcp.json` | MCP 外部工具服务器定义 | 修改后重启 |

`env.toml` 的路径可由环境变量 `FRONTIER_CONFIG` 覆盖，默认是工作目录下的 `env.toml`。机器人名称只来自 `.env` 的 `NICKNAME`：数组第一项是非空项中的默认显示名，全部非空项都作为全局唤醒词；某个群在数据库中配置了自定义唤醒词后，以该群配置为准。

`mcp.json` 每项必须提供 HTTP(S) `url`，`transport` 默认 `http`，也接受 `streamable_http`；支持 5–300 秒的 `startup_timeout_seconds` 和字符串 `headers`。旧的 stdio/SSE 接入以及 `command`、`args`、`env` 字段不再接受。

## 配置版本与校验

- 必须显式声明 `config_version = 2`，且只能是整数 `2`；其他值、缺失值或旧版分区会直接报错。
- 已退役的配置段：`[information]`、`[endpoint]`、`[llm_endpoints]`、`[function]`、`[message]`、`[database]`、`[image_memory]`、`[memory]`。
- 已移除的字段：`[models]` 下的 `*_model_endpoint`、`*_model_use_responses_api`、`paint_base_url`、`video_base_url`；`[key]` 下的模型 API key；`[providers.*]` 下的 `capabilities`、`provider` 和 `use_responses_api`；`[document_parsing]`、`[speech_recognition]`、`[speech_synthesis]` 下的 `provider`（改名为 `credentials_profile`，只用于选择提供密钥的供应商 profile）。模型 endpoint、协议和密钥应配置在对应供应商 profile。`[platforms.feishu]` 与 `[content_check]` 对应的功能已移除，旧配置里残留的这两个配置段会被忽略。
- 旧配置中的 `signal_model` / `signal_model_provider` / `signal_model_capabilities` 会映射为 `decision_*`，但新配置应直接使用 `decision_*`；同名新旧字段不能同时出现。
- 模型角色和其余配置段拒绝未知字段；供应商 profile 允许保留额外键。
- 启动时 `utils/configs.py` 校验成功后才替换运行时快照；校验失败不会用半成品配置继续运行。Dashboard 保存失败时会回滚旧文件并重新加载旧配置。

## 配置段总览

| 配置段 | 用途 | 详见 |
| --- | --- | --- |
| `[bot]` | 主 system prompt；`{name}` 会替换为当前唤醒词 | 本页 |
| `[models]` / `[providers.*]` | 模型角色、供应商 profile、密钥和协议 | 本页 |
| `[document_parsing]` | DeepSeek 文件 OCR 解析 | [DeepSeek 媒体能力](deepseek-media.md) |
| `[key]` | NASA、GitHub 等非模型服务密钥 | 本页 |
| `[features]` | 对话、绘画和视频功能开关 | 本页 |
| `[agent]` | 长文转图阈值、Fake-IP 媒体回退 | [消息媒体网络](message-media-network.md) |
| `[search]` | 独立网页搜索和读取工具 | [DeepSeek 联网](deepseek-web-search.md) |
| `[speech_recognition]` | DeepSeek 云端语音识别 | [DeepSeek 媒体能力](deepseek-media.md) |
| `[speech_synthesis]` | DeepSeek 云端语音合成 | [DeepSeek 媒体能力](deepseek-media.md) |
| `[agent_policy]` / `[auto_reply_policy]` / `[paint_policy]` | 访问名单与主动回复、绘画权限 | 本页 |
| `[limits]` | 调用预算、超时、限流 | [Agent 执行控制](agent-execution-controls.md) |
| `[sessions]` | QQ 持久会话的历史预算 | [QQ 持久会话](agent-sessions.md) |
| `[events]` | Milky 申请事件处理 | [Milky 申请事件](platform-events.md) |
| `[notifications]` | 内置任务与通知的初始目标群 | 本页 |
| `[storage]` | 历史查询窗口、图片与媒体 TTL | 本页 |
| `[debug]` | Agent 调试开关 | 本页 |
| `[dashboard]` | 管理面板密码与 JWT | 本页 |

## 模型与供应商

### 模型角色

`[models]` 按角色声明模型、供应商 profile 和模型能力，字段名为 `<角色>_model`、`<角色>_model_provider`、`<角色>_model_capabilities`：

| 角色 | 用途 |
| --- | --- |
| `basic` | 轻量辅助任务（如文档子代理、辅助 Agent） |
| `decision` | 回复门控、浏览器捕获意图等结构化判断 |
| `advanced` | 主对话 Agent；未配置时主 Agent 无法执行 |
| `daily_news` | 每日新闻的编辑与复核模型 |
| `paint` | 绘画模型，另有 `paint_size`、`paint_quality` |
| `video` | 视频模型，另有 `video_size`、`video_seconds` |

`*_model_capabilities` 描述该模型实际支持的多模态输入，可选值与别名：`text`、`vision`（别名 `image`）、`audio`、`video`、`file`（别名 `pdf`、`attachment`）。能力只配置在模型角色上，不能写在供应商 profile 里；供应商只描述接口。

`advanced_reasoning_effort` 设置主模型的思考强度，`basic_reasoning_effort` 用于辅助 Agent、文档子代理和逆地理编码。两项均可在 Dashboard 的「模型」页编辑。留空时使用供应商默认值，Responses 主模型则沿用 `[agent].reasoning_effort`（默认 `medium`）。

思考强度支持 `api_mode = "responses"` 和 `api_mode = "chat_completions"`。常用值为 `low`、`medium`、`high`、`xhigh`，仅为支持思考强度的模型和接口设置。对官方 DeepSeek 路由，HyperBot 将 `medium` 和 `xhigh` 分别映射为 `high` 和 `max`。

旧字段 `advanced_model_reasoning_effort` 和 `basic_model_reasoning_effort` 仍可读取。同一角色同时设置两个字段时，新名称优先。通过 Dashboard 保存「模型」后，配置文件会使用新名称。

`paint_size` 仍会被解析（默认 `"auto"`），但当前绘图请求固定使用 `size="auto"`，由图片模型结合提示词和参考图选择尺寸；`paint_quality` 会传给绘图接口。`video_size` 和 `video_seconds` 会传给视频接口。

### 供应商 profile

每个 `*_model_provider` 填写 `[providers.<name>]` 的名称。profile 的 `type` 决定底层 LangChain 适配器，`api_mode` 决定接口协议：

| `type` | 可用 `api_mode` | 说明 |
| --- | --- | --- |
| `openai` | `chat_completions`、`responses` | OpenAI-compatible 代理 |
| `google` | `generate_content` | Google Gemini |
| `anthropic` | `messages` | Anthropic Messages |
| `deepseek` | `chat_completions` | DeepSeek 专用适配器 |

省略 `api_mode` 时，新建 profile 按 `type` 回退默认协议（`openai` → `chat_completions`，`google` → `generate_content`，`anthropic` → `messages`，`deepseek` → `chat_completions`）；与内置同名 profile 未写 `api_mode` 时沿用内置值，例如 `[providers.openai]` 仍是 `responses`。内置 profile 名称始终可用：`openai`（默认 `responses`）、`google`、`anthropic`、`deepseek`、`deepseek_responses`（`openai + responses`，默认地址 `https://api.deepseek.com`）、`deepseek_anthropic`（`anthropic + messages`，默认地址 `https://api.deepseek.com/anthropic`）；同名 `[providers.<name>]` 覆盖默认值。

给内置 profile 设置密钥时只需写 `api_key`。默认 Decision 使用 `[providers.deepseek]`，默认独立搜索工具使用 `[providers.deepseek_responses]`。使用自定义网关时，可添加 `[providers.<name>]` 并设置 `type`、`base_url` 和 `api_key`，再让对应的 `*_model_provider` 指向该名称。

profile 常用字段：

- `base_url`、`api_key`：地址与密钥。OpenAI 适配器的地址依次取 profile 的 `base_url`、`OPENAI_API_BASE`、`OPENAI_BASE_URL` 和官方默认地址。官方能力判定与 SDK 使用同一个地址。
- `native_web_search`：设置 `false` 关闭托管搜索。支持搜索的官方 OpenAI Responses 模型默认启用 `web_search`，官方 DeepSeek Flash Responses 默认启用其搜索和网页读取工具。兼容 Responses 网关需显式设置 `true`。QQ 回复会编号展示结果里的 URL 引用，来源链接以文本单独投递。
- `native_tool_search`：官方 OpenAI Responses 路由仅在目录确认具备工具调用能力的 GPT-5.4 及后续模型上启用。设置 `false` 可关闭。兼容 Responses 网关确认支持后，可用 `true` 显式启用。Chat Completions 使用普通工具绑定。
- `responses_store`：OpenAI 适配器的 Responses 路由默认发送 `store=false`。设为 `true` 可为支持此选项的路由启用 API 端响应保存。官方 OpenAI 的无状态请求会请求加密推理项，用于后续工具调用。官方 DeepSeek Flash 托管搜索固定使用无状态模式。
- 官方 OpenAI profile 可设 `native_decisions = true` 并填写官方 `api_key`，同时在 `[models]` 中配置 `decision_model = "gpt-6-luna"` 和对应的 `decision_model_provider`。回复门控、平台事件判断和截图意图将使用 `/v1/decisions`。这是独立于 `api_mode` 的 Decision 开关，默认关闭。接口使用原生判断题、选择题和评分题，无需设置 `decision_extra_body`。[OpenAI Decisions API](https://developers.openai.com/api/docs/guides/decisions)
- `structured_output_method`：Decision 结构化策略，`auto`、`json_schema`、`function_calling`、`json_mode`、`text_json`，默认 `auto`。
- `decision_extra_body`：只作用于 decision 请求的 `extra_body`，例如关闭 DeepSeek 思考模式；旧字段 `signal_extra_body` 仍可读取，不能与 `decision_extra_body` 配置不同内容。
- `safety_check`：可选的 DeepSeek Chatbot API 安全审核参数，例如 `{ level = "high" }`。官方 DeepSeek profile 默认使用高强度审核。
- `title`：可选的会话标题参数，例如 `{ enabled = true }`。只会发送到官方 DeepSeek 路由。

DeepSeek 可分别配置为 `deepseek + chat_completions`、`openai + responses`、`anthropic + messages`。使用官方地址和 DeepSeek 模型时，HyperBot 会在三种协议中发送不含聊天正文的稳定 scope ID，用于服务端 KV cache、内容安全和调度隔离；兼容代理不会收到该扩展字段。视觉输入只应配置给实际支持视觉的模型，例如 `deepseek-flash`；`deepseek-v4-pro` 仍是纯文本模型。`*_model_capabilities` 必须与实际模型能力一致。

### 模型路由

- 显式 `*_model_provider` 优先；没有显式 provider 时按模型名前缀推断：`deepseek*`、`gemini-*`、`claude-*`，其余走 OpenAI-compatible。
- 供应商 profile 名称必须存在；未声明的内置名称按上表默认值可用，未知名称报错。
- `models/` 模型目录为命中的模型卡注入上下文窗口、输出上限和能力元数据；目录外模型按未知模型降级。
- 绘图使用 `type = "openai"`。官方端点通过 Images API 生成和编辑图片，独立于聊天协议。兼容网关按 `api_mode` 选择 Images API 或可重写顶层模型的 Responses `image_generation` 接口。
- 视频仅供支持 Videos API 的兼容网关使用。配置 `video_model` 和 `video_model_provider`，并将该 `type = "openai"` profile 指向网关。下载限制为 20 MB，响应中的结果链接通过公网地址校验后下载。 [官方停用公告](https://developers.openai.com/api/docs/deprecations)
- 每日新闻使用 `advanced_model` / `advanced_model_provider` 对应的主模型。归档预览和定时日报需要官方 DeepSeek Flash Responses，旧日报模型字段仍可解析。 [新闻插件运维](news-plugin.md)

## 其余配置段

### `[bot]`

`system_prompt` 是 QQ 会话的基础人设，和 `prompts/AGENTS.md` 的全局规范及 `prompts/qq.md` 的 QQ 说明共同组成系统提示词。`{name}` 按当前唤醒词注入。定时任务和日报归档使用各自的任务说明。

### `[features]`

`agent_enabled`、`paint_enabled`、`video_enabled` 控制对应入口。`qq_text_canary_enabled` 仅为旧配置兼容字段。

### `[key]`

这里保存 `nasa_api_key`（默认 `DEMO_KEY`）和 `github_pat`。模型密钥放在对应供应商 profile。

### `[agent]`

`message_image_text_threshold` 控制普通文字转长图的字符阈值（默认 500）；`message_media_fake_ip_fallback` 控制 Clash TUN Fake-IP DNS 回退，详见 [消息媒体网络](message-media-network.md)。

### `[speech_recognition]`

默认关闭。启用后网关通过才调用 DeepSeek `/v1/tools/asr`，API key 从 `[providers.<provider>]` 读取。QQ SILK 会在本地转换为 DeepSeek 支持的 WAV，原始音频仍保存为附件，转写文本进入历史。`max_audio_seconds`、`max_audio_bytes`、`max_audio_per_message` 和 `max_transcript_chars` 继续限制单条消息。

### `[speech_synthesis]`

默认关闭。启用后调用 DeepSeek `/v1/tools/tts`，`voice_id` 可选 `stella` 或 `tide`。服务端返回 WAV，系统转换为 QQ 所需的 SILK 语音工件。`max_text_chars` 默认 300，`max_audio_seconds` 默认 60，`timeout_seconds` 默认 180。`get_speech` 共用 `[paint_policy]` 权限，每个用户在 `rate_limit_window_seconds`（默认 600 秒）内可请求 `rate_limit_max_requests` 次（默认 3 次）。

### `[document_parsing]` 与 `[search]`

- 文件 OCR 的 `purpose` 可选 `text_recognition`（默认）或 `rich_layout_recognition`。`max_files_per_message`、`max_file_bytes`、`max_text_chars` 默认分别为 3、100 MiB 和 30000 字符。 [DeepSeek 媒体能力](deepseek-media.md)
- 网页正文的 `content_preview_chars` 默认 6000，`max_content_chars` 默认 24000。 [DeepSeek 联网](deepseek-web-search.md)

### 访问策略

- `[agent_policy]` 决定是否响应 QQ 消息。`whitelist_mode = true` 时，非零群必须在 `whitelist_group_list` 中，且用户必须在 `whitelist_person_list` 中。个人和群黑名单优先，且在白名单模式关闭时仍然生效。
- `[auto_reply_policy]`：控制群聊中未被 @/唤醒时是否走主动回复判断；`whitelist_mode = true` 时仅 `whitelist_group_list` 中的群参与，`blacklist_group_list` 优先。
- `[paint_policy]`：绘画和视频共用，覆盖 `/paint`、`/video` 命令和聊天 Agent 的媒体生成工具。黑名单优先；`whitelist_mode = true` 时用户号在 `whitelist_person_list` 中或群号在 `whitelist_group_list` 中即可使用，两份名单均为空时全部拒绝。可在 Dashboard 的“绘画与视频权限”中修改，保存后对后续调用生效。

### `[limits]`

绘画/视频限流与轮询、Agent 单轮超时、总任务超时、主图模型与工具调用预算。字段默认值与计数范围见 [Agent 执行控制](agent-execution-controls.md)。

| 用途 | 字段和默认值 |
| --- | --- |
| 绘画限流 | `paint_rate_limit_max_requests` 3 次，`paint_rate_limit_window_seconds` 600 秒 |
| 视频限流 | `video_rate_limit_max_requests` 1 次，`video_rate_limit_window_seconds` 900 秒 |
| 视频轮询 | `video_poll_interval_seconds` 15 秒，`video_poll_timeout_seconds` 900 秒 |
| Agent 超时 | `agent_llm_timeout_seconds` 900 秒，`agent_job_timeout_seconds` 3600 秒 |

### `[notifications]`

保存内置任务与通知的初始目标群。任务管理页的数据库映射在启动时会覆盖对应任务的推送群（`apod_everyday`、`earth_now`、`eq_usgs`、`daily_news`、`nrc_merchant_alert`），因此需要持久修改现有任务的接收群时应使用任务管理页。`announce_group_id` 用于更新完成后的重启通知；`earthquake_group_id` 也供 Wolfx 地震预警读取；`test_group_id` 仍会被解析，但当前没有读取者。

NASA 每日一图、每日地球、日报与 NRC 商人提醒的初始群分别由 `apod_group_id`、`earth_now_group_id`、`news_summary_group_id`、`nrc_merchant_group_id` 设置。目标群字段默认都是空列表。

### `[storage]`

`query_message_numbers` 限制回复网关判断和历史准备时查询的近期消息数，这些近期历史会作为背景随当前请求交给主 Agent（编排层再截取最近 50 条）；`image_enabled`、`image_ttl_days`、`media_ttl_days`、`max_inline_images`、`max_inline_media_bytes`、`image_auto_cleanup` 控制附件缓存、内联媒体预算和清理。附件与历史语义见 [消息、上下文与投递边界](message_flow.md)，数据库结构见 [数据库说明](database-identity-migration.md)。

### `[debug]`

`agent_debug_mode` 打开 Agent 调试输出；生产环境保持关闭。

### `[dashboard]`

首次部署时设置 `password`。省略 `jwt_secret` 时会生成随机密钥并保存在 `cache/.runtime_jwt_secret`。`jwt_expire_hours` 默认 24。

## Dashboard 保存与重载

Dashboard 设置页通过 `/api/dashboard/settings/*` 读写 `env.toml`：

- 保存串行处理并发更新，使用独立临时文件和原子替换；备份只保留最近 10 份。
- 敏感值在读取时脱敏；提交仍是脱敏值时保留原密钥。无实际变更时不创建备份。
- 保存后调用 `EnvConfig.reload()`，成功才递增 `EnvConfig.REVISION` 并切换运行时快照；失败时原子恢复旧文件并重新加载旧配置。
- 响应包含 `changed_fields`、`config_revision`、`activation` 和 `restart_required`。`env.toml` 管理的配置在后续请求/任务读取时生效，无需重启；模型相关子代理在下一轮按配置版本重建，已经开始的模型调用不会重新执行。
- `.env` 中的 NoneBot 启动参数和 `mcp.json` 的服务器列表仍需重启。
- 源文件省略的配置段仍可在 Dashboard 查看默认值。保存时只写入发生变化的设置，并按需创建配置段。

## 会话、工作区与 SOUL

- 每个群聊按 `group_id` 共享 `cache/sandbox/memory/group-{group_id}/SOUL.md`，每个私聊按 `user_id` 使用 `cache/sandbox/memory/dm-{user_id}/SOUL.md`。文件缺失时创建零字节文件，由 Agent 按稳定互动逐步记录局部人设和长期偏好；全局安全、权限和工具规范不会写入 SOUL。
- Agent 默认工作区是 `cache/sandbox/workspaces/{workspace_key}`；`/memory/{workspace_key}/` 映射到 memory 目录，`/skills/` 映射到仓库内置只读 Skill。
- 消息图片和上传文件保存在 `cache/sandbox/memory/{workspace_key}/...`；历史图片可重新注入为模型输入。
- 当前版本使用带 `group-` / `dm-` 前缀的会话目录；启动不会移动历史裸 `{id}` 目录，旧部署需先用包含迁移工具的历史版本完成升级。
- QQ 主 Agent 接收当前触发、区间背景和本人的近期轮次。`[sessions]` 设置回读上下文的预算，更早的聊天原文可通过聊天工具查询。详见 [QQ 持久会话](agent-sessions.md)。

## 诊断

配置错误会在启动或 Dashboard 保存时明确报错。常用检查：

```bash
uv run --locked python -c "from utils.configs import EnvConfig; print(EnvConfig.REVISION, EnvConfig.ADVAN_MODEL)"
uv run --locked pytest test/utils/configs_test.py -q
```

`[events]`、`[limits]`、`[sessions]` 的行为分别以对应专题文档和代码为准；修改配置前先阅读该段说明，不要依赖历史部署的旧字段。
