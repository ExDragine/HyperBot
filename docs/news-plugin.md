# News plugin operations

The production path is:

Clockwork -> `plugins.news.scheduler` -> main Agent with the `daily-news` Skill -> DeepSeek server-side search and URL reading -> evidence validation -> `news.db` archive -> render -> per-target delivery.

For a chat request, the main Agent loads the `daily-news` Skill on demand and returns the briefing in the current conversation. The scheduler and `/news preview` pin the same Skill when invoking the main Agent. The application validates verbatim quotes against DeepSeek's server-side search snippets and uses page metadata to supplement publication dates.

## Configuration

Copy `news.toml.example` to `news.toml` when overrides are needed. Configure the main model with the official DeepSeek Flash Responses route. The scheduler uses the main model, its profile, and its credentials. Existing news configuration files can still contain `sources` and the former dedicated-model options. [DeepSeek web tools](deepseek-web-search.md)

The archives require sources with a usable date inside the edition window and a snippet that supports the quote. Unsupported main model routes fail before generating an archived edition.

## Delivery semantics

A report is archived before delivery. Each QQ group has its own state. `sent` is terminal; `failed` may be retried explicitly. A previously sent edition is reused without sending it again. Clockwork records a failed task result when any requested target lacks a confirmed `sent` state.

## Recovery and retention

The generation lease and `generation_timeout` cover research and all editing attempts. Each source article may support only one story across both sections, including supplementary citations. If an Agent draft repeats a source, contains invalid evidence, or has too few supported stories, the editor receives the validation error and reuses the collected, dated evidence to produce a corrected draft. `max_generation_attempts` bounds the total number of drafts, including the initial Agent draft; it does not rerun searches on each editing attempt. Valid drafts are archived immediately. Too few dated sources, an exhausted draft budget, or a timeout marks the report failed, and the next invocation may recollect. Retention removes inactive reports and delivery rows after `retention_days`.

## Operator commands

```
/news latest
/news status
/news preview
/news retry <group-id>
```

The `/news preview` command uses a separate preview archive and does not deliver to subscribed groups.
