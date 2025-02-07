"""LangGraph topology builder for the conversational agent.

Constructs a directed graph with conditional routing from the classifier node
to greeting, knowledge retrieval (RAG), appointment booking, and apology branches.

Graph topology::

    classifier → [conditional edges]
      ├─ "1" → greeter → END
      ├─ "2" → retriever → rag → END
      ├─ "3" → booking → END
      └─ "0" → apology → END

The booking node is also entered directly when the state contains partial
``user_details`` with missing required fields, bypassing the classifier.
"""

from langchain_groq import ChatGroq
from langgraph.graph import END, StateGraph

from groqflow.config import GROQ_MODEL, LLM_MAX_TOKENS, LLM_TEMPERATURE, RETRIEVAL_K
from groqflow.config_loader import TenantConfig, get_user_details_model
from groqflow.knowledge_base import KnowledgeBase
from groqflow.nodes.apology import ApologyResponder
from groqflow.nodes.booking import BookingNode
from groqflow.nodes.classifier import ClassifierNode
from groqflow.nodes.greeter import GreeterNode
from groqflow.nodes.rag import RagNode
from groqflow.nodes.retriever import RetrieverNode
from groqflow.state import AgentState


def build_graph(
    tenant: TenantConfig,
    knowledge_base: KnowledgeBase,
    model: str = GROQ_MODEL,
) -> object:
    """Build and compile the LangGraph agent for a tenant.

    Args:
        tenant: The tenant configuration (used for prompts and booking schema).
        knowledge_base: The tenant's FAISS knowledge base instance.
        model: Groq model identifier (default from environment).

    Returns:
        A compiled LangGraph ``CompiledGraph`` ready for ``.invoke()``.
    """
    llm = ChatGroq(
        model=model,
        temperature=LLM_TEMPERATURE,
        max_tokens=LLM_MAX_TOKENS,
    )
    user_details_model = get_user_details_model(tenant)
    classifier = ClassifierNode(llm=llm, tenant=tenant)
    greeter = GreeterNode(llm=llm, tenant=tenant)
    retriever = RetrieverNode(knowledge_base, RETRIEVAL_K)
    rag = RagNode(llm=llm)
    booking = BookingNode(llm=llm, tenant=tenant)
    apology = ApologyResponder()

    def route(state: AgentState) -> str:
        """Determine the next node based on classifier output and booking state.

        If the user has partially filled booking details with missing required
        fields, route directly to booking (continuing the slot-filling loop).
        Otherwise, route based on the classifier's category output.
        """
        user_details = state.get("user_details") or {}
        if (
            not state.get("booking_confirmed")
            and user_details
            and any(
                not user_details.get(field)
                or (
                    isinstance(user_details.get(field), str)
                    and not user_details[field].strip()
                )
                for field in user_details_model.get_required_field_names()
            )
        ):
            return "booking"

        choice = state.get("choice")
        category = getattr(choice, "content", "").strip()
        return {"1": "greeter", "2": "retriever", "3": "booking"}.get(
            category, "apology"
        )

    graph = StateGraph(AgentState)

    graph.add_node("classifier", classifier)
    graph.add_node("greeter", greeter)
    graph.add_node("retriever", retriever)
    graph.add_node("rag", rag)
    graph.add_node("booking", booking)
    graph.add_node("apology", apology)

    graph.set_entry_point("classifier")
    graph.add_conditional_edges("classifier", route)
    graph.add_edge("retriever", "rag")
    graph.add_edge("greeter", END)
    graph.add_edge("rag", END)
    graph.add_edge("booking", END)
    graph.add_edge("apology", END)

    return graph.compile()
