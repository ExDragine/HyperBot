# OpenAI 媒体和平台接口

通过 `[providers.openai]` 配置官方 API 密钥。OpenAI 语音和平台操作使用官方端点。绘图另外支持已有的兼容网关。

## QQ 语音

在 `env.toml` 中按需开启转写或合成。

```toml
[speech_recognition]
enabled = true
backend = "openai"
credentials_profile = "openai"
model = "gpt-transcribe"

[speech_synthesis]
enabled = true
backend = "openai"
credentials_profile = "openai"
model = "gpt-realtime-2.1-mini"
voice_id = "marin"
```

QQ 语音的 WAV、Ogg 和 SILK 输入会在验证大小及音频时长后送往 Audio Transcriptions。合成走 Realtime WebSocket，等待完整音频，再转换为 QQ SILK。系统会在语音前发送「AI 合成语音」提示。省略 `backend` 时沿用 DeepSeek。OpenAI 后端默认使用上述模型和音色，也可显式配置旧版 `gpt-4o-mini-tts` 与 `alloy`，旧接口计划在 2027 年 1 月 6 日停用。

[OpenAI 文件转写](https://developers.openai.com/api/docs/guides/speech-to-text) · [OpenAI 语音合成](https://developers.openai.com/api/docs/guides/text-to-speech)

## QQ 绘图

将 `[models].paint_model` 设为 `gpt-image-2.6-flare` 或 `gpt-image-2.6-sunburst`，并将 `paint_model_provider` 指向官方 profile。用户可直接在聊天里要求指定格式或标注编辑，由 `get_paint` 传入 `options`。普通绘图和 `/paint` 默认发送 PNG。

| 选项 | 用法 |
| --- | --- |
| `output_format` | PNG、JPEG、WebP、GIF、SVG。SVG 只支持生成。GIF 通过 Milky 图片消息发送，私聊用 `send_private_message`，群聊用 `send_group_message`。SVG 以 QQ 文件发送。 |
| `complexity` | GPT-Image-2.6 可设为 `low`、`medium` 或 `high`，与 `paint_quality` 分开控制。 |
| `frame_count`、`frame_duration_ms` | GIF 帧数 2–16，单帧时长 20–2000 毫秒。 |
| `use_markup` | GPT-Image-2.5/2.6 编辑时，将最后一张参考图作为标注层。标注层应与第一张原图同尺寸，并包含透明区域。 |

[OpenAI 图像生成与编辑](https://developers.openai.com/api/docs/guides/image-generation)

## 平台操作

在仓库根目录调用运维命令。若使用自定义官方 profile，在子命令前加 `--profile <名称>`。文本可用 `--text` 传入，结构化或含文件引用的 Responses 输入可用 `--json <路径>` 传入。成功结果输出到标准输出，可交给下游脚本处理。

```sh
uv run --locked python -m scripts.openai_platform --help
uv run --locked python -m scripts.openai_platform conversation-create
uv run --locked python -m scripts.openai_platform respond --model gpt-6.2-sol --conversation-id CONVERSATION_ID --text '概括当前对话'
uv run --locked python -m scripts.openai_platform background-start --model gpt-6.2-sol --completion-window 2h --json input.json
uv run --locked python -m scripts.openai_platform response-get RESPONSE_ID
uv run --locked python -m scripts.openai_platform batch-upload requests.jsonl
uv run --locked python -m scripts.openai_platform batch-create INPUT_FILE_ID --endpoint /v1/responses
uv run --locked python -m scripts.openai_platform batch-get BATCH_ID
uv run --locked python -m scripts.openai_platform batch-download OUTPUT_FILE_ID > results.jsonl
uv run --locked python -m scripts.openai_platform embed --text '用于检索的文字'
uv run --locked python -m scripts.openai_platform moderate --json moderation-input.json
```

命令还支持查看和删除会话、分页读取会话项，以及取消后台响应和批处理。批处理创建失败后可复用上传返回的文件 ID。后台响应凭响应 ID 查询，有时限的任务也可选择 `8h`；标准请求默认使用 `soonest`。Python 调用方可使用 `async with openai_platform(profile_name)`，通过返回的客户端执行同一组操作。

平台命令仅发送显式提供的数据。使用云端会话时，OpenAI 会保留会话项。后台任务会为执行和结果查询临时保存数据。QQ 聊天记录继续由应用的 SQLite 会话管理。

[会话状态](https://developers.openai.com/api/docs/guides/conversation-state) · [后台模式](https://developers.openai.com/api/docs/guides/background) · [Batch](https://developers.openai.com/api/docs/guides/batch) · [Embeddings](https://developers.openai.com/api/docs/guides/embeddings) · [Moderation](https://developers.openai.com/api/docs/guides/moderation)
