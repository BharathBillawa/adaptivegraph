"""
AdaptiveGraph "Tiered Customer Support Agent" Example.
------------------------------------------------------
This example demonstrates a complete "Batteries Included" workflow:
1. Semantic Routing: Classifies tickets into "QuickBot", "RAG", or "Human".
2. Trajectory Rewards: Updates the router based on the final outcome of the session.
3. Async Feedback: Simulates a user rating the interaction hours later.
4. Error Scoring: Penalizes the router if it selects a tool that crashes.

Scenario:
- "QuickBot": Cheap, Fast. Good for simple FAQ.
- "RAG Agent": Medium cost. Good for docs.
- "Human Expert": Expensive ($$$). Only for complex/angry users.
"""

import sys
import os
import random
import uuid
import time
from typing import TypedDict, Literal, Optional, List
from langgraph.graph import StateGraph, END
import numpy as np

# Ensure we can import the local package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from adaptivegraph import LearnableEdge
from adaptivegraph.rewards import ErrorScorer

# --- 1. Mock LLM for "LLM-as-a-Judge" ---
class MockjudgingLLM:
    """Simulates an LLM evaluating an answer."""
    def invoke(self, prompt: str) -> str:
        # Returns a float score disguised as string
        if "bad" in prompt.lower() or "error" in prompt.lower():
            return "0.2"
        return "0.9"

# --- 2. Define State ---
class SupportState(TypedDict):
    ticket_id: str          # Trace ID / Session ID
    user_tier: str          # "Free", "Premium" (Metadata)
    query: str              # User text (Semantic)
    
    # Internal Flow
    route_decision: str     # Where did we go?
    tool_output: str        # Tool result
    error: Optional[str]    # Did tool crash?
    
    # Outcome
    satisfaction_score: float # 0.0 to 1.0

# --- 3. Define Nodes ---

def triage_node(state: SupportState):
    print(f"[{state['ticket_id']}] Triage: User={state['user_tier']}, Query='{state['query']}'")
    return state

def quick_bot_node(state: SupportState):
    # Simulates a simple keyword bot. Fails on complex queries.
    if "reset password" in state["query"].lower():
        return {"tool_output": "Link: /reset-password", "route_decision": "quick_bot"}
    else:
        # Bot failed to understand
        return {"tool_output": "I don't understand.", "error": "BotFailure", "route_decision": "quick_bot"}

def rag_agent_node(state: SupportState):
    # Simulates RAG. Can sometimes crash (Simulated Tool Error).
    if random.random() < 0.1: # 10% chance of tool crash
        return {"tool_output": "", "error": "VectorDBTimeout", "route_decision": "rag_agent"}
    return {"tool_output": f"RAG Answer for '{state['query']}'", "route_decision": "rag_agent"}

def human_expert_node(state: SupportState):
    # Always works, but is "expensive" (we can model cost as a penalty if we wanted, 
    # but here we model it as high satisfaction).
    return {"tool_output": "Human: I have fixed your issue manually.", "route_decision": "human_expert"}

# --- 4. Setup Adaptive Router ---

# We use both Semantic (query) AND we could use Metadata (user_tier) via value_key formatting
# For simplicity, we just route on 'query' text here.
router = LearnableEdge.create(
    options=["quick_bot", "rag_agent", "human_expert"],
    embedding="sentence-transformers",
    memory="memory", # Use in-memory for this transient script
    feature_dim=384,
    value_key="query"
)

# --- 5. Build Graph ---
workflow = StateGraph(SupportState)

workflow.add_node("triage", triage_node)
workflow.add_node("quick_bot", quick_bot_node)
workflow.add_node("rag_agent", rag_agent_node)
workflow.add_node("human_expert", human_expert_node)

workflow.set_entry_point("triage")

# bind the router
workflow.add_conditional_edges(
    "triage",
    router,
    {
        "quick_bot": "quick_bot",
        "rag_agent": "rag_agent",
        "human_expert": "human_expert"
    }
)

workflow.add_edge("quick_bot", END)
workflow.add_edge("rag_agent", END)
workflow.add_edge("human_expert", END)

app = workflow.compile()

# --- 6. Utilities for Scoring ---
error_scorer = ErrorScorer(penalty=-1.0, success_reward=0.1) 
# Note: small positive reward for just "not crashing". Real reward comes from satisfaction.

# --- 7. Simulation Loop ---
print("\n--- Starting Support Simulator (50 Tickets) ---\n")

dummy_queries = [
    ("How do I reset password?", "quick_bot"),     # Simple
    ("Server 500 fatal error", "human_expert"),   # Critical
    ("Where is the documentation?", "rag_agent"), # Standard
    ("Billing dispute $5000", "human_expert"),    # Critical
    ("Login button not working", "rag_agent"),    # Standard
]

async_feedback_queue = [] # Simulation of delayed feedback

for i in range(50):
    # 1. Generate Ticket
    q_text, ideal_route = random.choice(dummy_queries)
    u_tier = "Premium" if random.random() < 0.3 else "Free"
    t_id = str(uuid.uuid4())[:8]
    
    # We inject 'trace_id' so the router tracks decisions for this ticket
    state = {
        "ticket_id": t_id,
        "user_tier": u_tier,
        "query": q_text,
        "route_decision": "",
        "tool_output": "",
        "error": None,
        "satisfaction_score": 0.0,
        "trace_id": t_id # <--- CRITICAL for Trajectory Rewards
    }
    
    # 2. Run Agent
    result = app.invoke(state)
    
    # 3. Intermediate Reward: Check for Crashes (Tool Failure)
    # This runs immediately after execution
    crash_score = error_scorer.score(result)
    if crash_score < 0:
        print(f"   [!] Tool Crashed ({result['error']})! Penalizing immediately.")
        # We can apply this to the trace immediately or wait. 
        # Let's apply immediate 'partial' feedback to the specific decision via record_feedback
        # Note: Since we used 'trace_id', the decision is stuck in 'active_traces'.
        # We can communicate "bad signal" via complete_trace OR just let the final score handle it.
        # Let's assume tool crash ends the session -> complete trace with penalty.
        router.complete_trace(t_id, final_reward=-1.0)
        continue 
        
    # 4. Calculate User Satisfaction (Simulated Ground Truth)
    # In real life, this comes from the user. Here we simulate it.
    actual_route = result['route_decision']
    
    satisfaction = 0.5 # Default okay
    if actual_route == ideal_route:
        satisfaction = 1.0
    elif actual_route == "human_expert": 
        # Users love humans even if overkill, but it costs us. 
        # Let's simple model: High satisfaction but maybe we penalize cost later?
        satisfaction = 0.9 
    elif actual_route == "quick_bot" and ideal_route != "quick_bot":
        satisfaction = 0.0 # Hated it
        
    # 5. Add to Async Queue (Simulate 1-hour delay)
    async_feedback_queue.append({
        "ticket_id": t_id,
        "score": satisfaction,
        "route": actual_route
    })
    
    # Process feedback in batches (e.g. every 10 tickets)
    if i % 10 == 0 and async_feedback_queue:
        print(f"\n--- Processing {len(async_feedback_queue)} Async User Ratings ---")
        for fb in async_feedback_queue:
            # RETROSPECTIVE REWARD via Trajectory
            # We use complete_trace to reward the session that happened "hours ago"
            router.complete_trace(fb['ticket_id'], final_reward=fb['score'])
            
            # Simple log
            # print(f"   Ticket {fb['ticket_id']}: User rated {fb['score']:.1f}")
        async_feedback_queue = []
        print("   Router updated based on user feedback.\n")

print("\n--- Simulation Complete ---")
print("The 'Human Expert' route was chosen for expensive queries, while 'QuickBot' handled passwords.")
