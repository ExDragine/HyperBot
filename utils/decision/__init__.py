"""Platform-neutral decision providers used by gates and routers."""

from utils.decision_llm import DecisionLLM, decision_structured

from .llm import DecisionEnvelope, LLMDecisionProvider
from .models import DecisionProvider, DecisionQuestions, DecisionResult, DecisionState
from .reply_gate import REPLY_GATE_QUESTIONS, ReplyGateScore, build_reply_gate_state, score_reply_gate

__all__ = [
    "DecisionProvider",
    "DecisionQuestions",
    "DecisionResult",
    "DecisionState",
    "DecisionEnvelope",
    "LLMDecisionProvider",
    "DecisionLLM",
    "decision_structured",
    "REPLY_GATE_QUESTIONS",
    "ReplyGateScore",
    "build_reply_gate_state",
    "score_reply_gate",
]
