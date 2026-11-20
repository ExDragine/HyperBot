"""Exercise the QQ history budget with real Deep Agents outside the test stubs."""

# ruff: noqa: S101

import subprocess
import sys
from pathlib import Path


def test_real_qq_graph_reads_bounded_history_from_each_request(tmp_path):
    (tmp_path / "env.toml").write_text('''
config_version = 2
[models]
basic_model = "gpt-4o-mini"
advanced_model = "gpt-4o"
[sessions]
history_max_tokens = 128
''', encoding="utf-8")
    script = r'''
import asyncio
import os
import sys
sys.path.insert(0, sys.argv[1])
os.environ.setdefault('NICKNAME', '["Test"]')

from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage
from utils.agent_protocol import ConversationRef
from utils.agents import cognitive
from utils.agents.session_context import QQ_TRIGGER_ID_PREFIX
from utils.database import MessageDatabase

observed = []
class Model(FakeMessagesListChatModel):
    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, *args, **kwargs):
        observed.append([(item.type, item.content) for item in messages])
        return super()._generate(messages, *args, **kwargs)

cognitive.create_llm = lambda **kw: Model(responses=[AIMessage(content='final')])

async def main():
    MessageDatabase()
    agent = cognitive.FrontierCognitive()

    async def turn(number, history):
        current = {'role': 'user', 'content': f'current {number}', 'id': f'{QQ_TRIGGER_ID_PREFIX}{number}'}
        result = await agent.chat_agent(
            [*history, current], user_id='10', user_name='test', thread_id_override='same-execution-thread',
            group_id=20, conversation=ConversationRef('qq', '99', 'group', '20'),
            allow_local_tools=False, allow_native_web_search=False, enable_acp_subagents=False, include_messages=True,
        )
        assert result['status'] == 'success', result
        assert [m.content for m in result['response']['messages'] if m.type == 'ai'] == ['final'], result

    await turn(10, [
        {'role': 'user', 'content': 'old question ' + 'x ' * 1000},
        {'role': 'assistant', 'content': 'old answer'},
        {'role': 'user', 'content': 'recent question', 'id': 'recent-user'},
        {'role': 'assistant', 'content': 'confirmed answer', 'id': 'recent-reply'},
    ])
    await turn(11, [])
    assert [item for item in observed[0] if item[0] != 'system'] == [
        ('human', 'recent question'), ('ai', 'confirmed answer'), ('human', 'current 10'),
    ], observed[0]
    assert [item for item in observed[1] if item[0] != 'system'] == [('human', 'current 11')], observed[1]

asyncio.run(main())
'''
    result = subprocess.run(  # noqa: S603
        [sys.executable, "-c", script, str(Path(__file__).resolve().parents[2])],
        cwd=tmp_path, capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
