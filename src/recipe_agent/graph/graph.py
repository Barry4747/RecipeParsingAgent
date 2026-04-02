import structlog
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from recipe_agent.graph.state import AgentState
from recipe_agent.graph.nodes import (
    node_parse,
    node_translate,
    node_human_review,
    node_save,
)

log = structlog.get_logger()


def route_after_parse(state: AgentState) -> str:
    if state.get("parse_error"):
        if state.get("retry_count", 0) >= 3:
            log.error("graph.parse.max_retries")
            return "end"
        return "parse"
    return "translate"


def route_after_review(state: AgentState) -> str:
    decision = state.get("human_decision")
    if decision == "save":
        return "save"
    if decision == "edit":
        return "parse"
    return "end"


def build_graph() -> StateGraph:
    builder = StateGraph(AgentState)

    builder.add_node("parse", node_parse)
    builder.add_node("translate", node_translate)
    builder.add_node("human_review", node_human_review)
    builder.add_node("save", node_save)

    builder.add_edge(START, "parse")

    builder.add_conditional_edges("parse", route_after_parse, {
        "translate": "translate",
        "parse": "parse",
        "end": END,
    })

    builder.add_edge("translate", "human_review")

    builder.add_conditional_edges("human_review", route_after_review, {
        "save": "save",
        "parse": "parse",
        "end": END,
    })

    builder.add_edge("save", END)

    checkpointer = MemorySaver()
    return builder.compile(
        checkpointer=checkpointer,
        interrupt_before=["human_review"],
    )


graph = build_graph()

from recipe_agent.graph.nodes import node_save_migration

def build_migration_graph() -> StateGraph:
    builder = StateGraph(AgentState)

    builder.add_node("parse", node_parse)
    builder.add_node("translate", node_translate)
    builder.add_node("human_review", node_human_review)
    builder.add_node("save", node_save_migration)

    builder.add_edge(START, "parse")

    builder.add_conditional_edges("parse", route_after_parse, {
        "translate": "translate",
        "parse": "parse",
        "end": END,
    })

    builder.add_edge("translate", "human_review")

    builder.add_conditional_edges("human_review", route_after_review, {
        "save": "save",
        "parse": "parse",
        "end": END,
    })

    builder.add_edge("save", END)

    checkpointer = MemorySaver()
    return builder.compile(
        checkpointer=checkpointer,
        interrupt_before=["human_review"],
    )


migration_graph = build_migration_graph()