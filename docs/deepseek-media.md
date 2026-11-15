# DeepSeek 媒体能力

Frontier 的语音识别、语音合成和文档 OCR 使用 DeepSeek 的托管 API，不再加载本地 SenseVoice 或 MOSS-TTS 模型。

在 `env.toml` 中启用：

```toml
[providers.deepseek]
type = "deepseek"
api_mode = "chat_completions"
base_url = "https://api.deepseek.com"
api_key = "..."

[speech_recognition]
enabled = true
credentials_profile = "deepseek"

[speech_synthesis]
enabled = true
credentials_profile = "deepseek"
voice_id = "stella"
```

语音识别调用 `POST /v1/tools/asr`，支持 WAV 和 Ogg Opus。QQ SILK 会先在本地转换为 WAV，模型识别本身在 DeepSeek 服务端完成。语音合成调用 `POST /v1/tools/tts`，收到 WAV 后转换为 QQ 所需的 SILK 工件。

收到文件消息时，启用 `[document_parsing]` 后，文件会在网关通过并落盘索引后自动进入 DeepSeek OCR。当前支持 PDF、Word、PowerPoint 和常见图片，OCR 文本按页截断后注入本轮上下文，同时保留原文件路径供 Agent 继续读取。`text_recognition` 适合纯文字文档，`rich_layout_recognition` 会尝试保留图表和图片信息，但会消耗更多输入资源。

需要按需重新识别工作区文件时，Agent 也可以使用 `deepseek_ocr` 工具。它调用 `POST /v1/tools/ocr`，支持两种解析模式，文件必须先存在于当前 workspace 或 `/memory/...` 挂载路径。

QQ SILK 的本地编解码只承担格式转换，媒体请求失败只影响当前转写、朗读或 OCR，不会阻塞普通文本回复。
