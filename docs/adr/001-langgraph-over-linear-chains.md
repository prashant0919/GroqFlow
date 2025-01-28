# ADR-001: LangGraph Over Linear LangChain Chains

**Status:** Accepted  
**Date:** Day 01 — System Architecture Specification  
**Author:** Prashant Bastola  
**Deciders:** Prashant Bastola

## Context

GroqFlow requires two distinct conversational flows:
1. **Knowledge Q&A**: User asks a question → retrieve context from FAISS → generate a grounded response
2. **Appointment Booking**: User requests an appointment → extract provided details → identify missing fields → loop back to ask for remaining fields → confirm when complete

A straightforward LangChain sequential chain (`prompt | llm | parser`) works for the Q&A flow, but the booking flow requires **stateful cycles** — the agent must repeatedly loop back to collect missing information across multiple user turns.

## Decision

Use **LangGraph** (`StateGraph`) as the orchestration framework instead of linear LangChain chains.

## Rationale

### Why not a simple LangChain chain?

Linear chains execute a fixed sequence of steps from input to output. They do not support:
- **Conditional branching**: Routing to different processing paths based on intent classification
- **Cyclic execution**: Looping back to ask for additional booking details until all required fields are present
- **Shared mutable state**: Accumulating extracted booking details across multiple turns without external state management

### The graph topology

```
classifier → [conditional edges]
  ├─ "1" → greeter → END
  ├─ "2" → retriever → rag → END
  ├─ "3" → booking → END
  └─ "0" → apology → END
```

The booking node internally manages the slot-filling loop state through `user_details` in `AgentState`. Across user turns, the graph is re-invoked with the accumulated state, so the slot-filling naturally continues from where it left off.

## Consequences

### Positive
- Each node is independently testable (receives `AgentState`, returns `AgentState`)
- Graph topology is declarative and visible in `build_graph()`
- State schema is typed via `TypedDict`, reducing runtime errors
- Adding new intent routes requires adding a node and a conditional edge — no refactoring of existing nodes

### Negative
- LangGraph is a newer library with less community documentation than core LangChain
- Developers must understand the `StateGraph` / `add_messages` reducer pattern
- State must be persisted externally (SQLite) between turns
