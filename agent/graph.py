"""The assistant's brain: a LangGraph graph around Gemini, with tools.

    START -> assistant --(tool calls?)--> tools -> assistant -> ... -> END

LiveKit sends the whole conversation (chat context) on every turn through
`langchain.LLMAdapter`, so this graph is stateless: messages in -> reply out.

What gets SPOKEN is explicit: only text passed to `get_stream_writer()` reaches
TTS (LLMAdapter runs with stream_mode="custom"). Tool calls and tool results
stay internal, so the assistant never reads raw tool output aloud.

Try it as a text chat: uv run agent/graph.py
"""

import ast
import operator
from datetime import datetime
from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage, SystemMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.config import get_stream_writer
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

MODEL = "gemini-3.5-flash-lite"  # ~1s first token; 3.5-flash / 3.8-flash measured 6-50s (too slow for voice)

# Replies are spoken by TTS, so: short, plain words, no markdown/emoji/lists.
SYSTEM_PROMPT = """You are a friendly, helpful voice assistant.
Answer questions clearly and briefly: usually one to three short sentences.
Your words will be spoken aloud, so use plain conversational language.
Never use markdown, bullet points, emojis, code blocks, or special symbols.
Write numbers and abbreviations the way they should be said.
If a question is ambiguous, ask one short clarifying question.
Use the get_current_datetime tool for any question about today's date or the time.
Use the calculator tool for any arithmetic instead of computing it yourself."""


# ---- Tools --------------------------------------------------------------------
# @tool turns a function into a schema (name + docstring + typed args) that Gemini
# can choose to call. The docstring is what the model reads, so make it precise.

@tool
def get_current_datetime() -> str:
    """Get the current local date, weekday and time."""
    return datetime.now().strftime("%A, %d %B %Y, %I:%M %p")


_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
    ast.Pow: operator.pow, ast.Mod: operator.mod, ast.FloorDiv: operator.floordiv, ast.USub: operator.neg,
}


def _eval(node: ast.AST) -> float:
    # Walk the parsed expression and allow only numbers and arithmetic operators.
    # Never use eval() on model output: it would run arbitrary Python.
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        if isinstance(node.op, ast.Pow) and abs(_eval(node.right)) > 100:
            raise ValueError("exponent too large")
        return _OPS[type(node.op)](_eval(node.left), _eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.operand))
    raise ValueError("unsupported expression")


@tool
def calculator(expression: str) -> str:
    """Evaluate an arithmetic expression, e.g. '(12.5 * 4) / 3' or '2 ** 10'. Supports + - * / // % ** and parentheses."""
    try:
        result = _eval(ast.parse(expression, mode="eval").body)
    except (ValueError, SyntaxError, ZeroDivisionError, OverflowError) as e:
        return f"error: {e}"
    return f"{result:.10g}"


TOOLS = [get_current_datetime, calculator]


# ---- Graph --------------------------------------------------------------------

class State(TypedDict):
    # add_messages = reducer: new messages are appended to the list, not replacing it.
    messages: Annotated[list[BaseMessage], add_messages]


def build_graph(model: str = MODEL):
    llm = ChatGoogleGenerativeAI(
        model=model,
        thinking_config={"thinking_level": "low"},  # voice needs speed; skip long reasoning
    ).bind_tools(TOOLS)  # tell Gemini which tools exist (their JSON schemas)

    async def assistant(state: State):
        say = get_stream_writer()  # text given to say() is streamed to TTS as it arrives
        reply = None
        # Prepend our system prompt; LiveKit's chat context holds the rest of the history.
        async for chunk in llm.astream([SystemMessage(SYSTEM_PROMPT), *state["messages"]]):
            if chunk.text:
                say(chunk.text)  # speak tokens immediately (first sentence plays early)
            reply = chunk if reply is None else reply + chunk  # merge chunks (incl. tool calls)
        return {"messages": [reply]}

    builder = StateGraph(State)
    builder.add_node("assistant", assistant)
    builder.add_node("tools", ToolNode(TOOLS))  # runs whatever tool calls the model asked for
    builder.add_edge(START, "assistant")
    # If the reply contains tool calls -> "tools", else -> END (answer is done).
    builder.add_conditional_edges("assistant", tools_condition)
    builder.add_edge("tools", "assistant")  # feed tool results back so the model can answer
    return builder.compile()


if __name__ == "__main__":
    import asyncio
    import time

    from dotenv import load_dotenv
    from langchain_core.messages import AIMessage, HumanMessage

    load_dotenv(".env.local")

    async def chat():
        graph = build_graph()
        history: list[BaseMessage] = []  # we keep the memory here (LiveKit does it in the agent)
        while (text := input("\nyou> ").strip()) not in ("", "exit"):
            history.append(HumanMessage(text))
            start, first, reply = time.perf_counter(), None, ""
            print("bot> ", end="", flush=True)
            # stream_mode="custom" yields exactly what the nodes passed to say(); LLMAdapter uses the same.
            async for text in graph.astream({"messages": history}, stream_mode="custom"):
                first = first or time.perf_counter()
                print(text, end="", flush=True)
                reply += text
            history.append(AIMessage(reply))
            print(f"\n   (first token {first - start:.2f}s, total {time.perf_counter() - start:.2f}s)")

    asyncio.run(chat())
