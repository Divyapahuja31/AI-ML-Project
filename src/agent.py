"""
Agentic AI Lending Decision Support System — Milestone 2.

Implements a two-node LangGraph workflow:

  retrieve_policies → generate_report → END

Node 1 (retrieve_policies): Runs a RAG query against the policy vector store
  to surface relevant lending guidelines given the borrower's risk profile.

Node 2 (generate_report): Calls a Groq-hosted open-source LLM (LLaMA-3.3-70B)
  with the borrower data, ML risk score, SHAP feature attributions, and the
  retrieved policy context to produce a structured lending assessment.

If no Groq API key is provided OR if the LLM call fails, the agent falls back
to a deterministic template-based report so the application always works.

Usage
-----
Set ``GROQ_API_KEY`` as an environment variable or pass it explicitly.
Free API key: https://console.groq.com
"""

import json
import os
import re
from typing import Any, Dict, List, Optional

from typing_extensions import TypedDict
from src.explain_rules import generate_lending_assessment


# ─── LangGraph state schema ───────────────────────────────────────────────────

class LendingState(TypedDict):
    # ── Inputs ────────────────────────────────────────────────────────────────
    borrower_profile: Dict[str, Any]   # Raw form values
    risk_score:       float            # ML default probability [0, 1]
    risk_label:       str              # "High Risk" | "Low Risk"
    top_features:     List[tuple]      # [(feature_name, shap_value), ...]
    # ── Intermediate ──────────────────────────────────────────────────────────
    policy_chunks:    List[Dict]       # Retrieved policy snippets
    # ── Output ────────────────────────────────────────────────────────────────
    report:           Dict[str, str]


# ─── Node 1: Policy retrieval ─────────────────────────────────────────────────

def node_retrieve_policies(state: LendingState) -> LendingState:
    """RAG: Retrieve the most relevant lending policy context."""
    try:
        from src.rag import query_policies

        query = (
            f"lending decision guidelines for {state['risk_label'].lower()} borrower "
            f"probability of default {state['risk_score']:.0%} "
            f"delinquency debt ratio credit utilization"
        )
        chunks = query_policies(query, k=4)
    except Exception:
        chunks = []

    return {**state, "policy_chunks": chunks}


# ─── Node 2: LLM report generation ───────────────────────────────────────────

def node_generate_report(state: LendingState, llm: Any) -> LendingState:
    """LLM: Produce a structured lending assessment report."""
    from langchain_core.messages import HumanMessage

    # Format policy context
    policy_text = "\n\n".join(
        f"[Source: {c.get('source', 'policy')}]\n{c['content']}"
        for c in state.get("policy_chunks", [])
    ) or "No specific policy documents retrieved."

    # Format SHAP feature list
    feature_text = "\n".join(
        f"  {i+1}. {feat}: SHAP {val:+.4f}"
        for i, (feat, val) in enumerate(state["top_features"][:5])
    )

    # Format borrower profile
    borrower_text = "\n".join(
        f"  • {k.replace('_', ' ').title()}: {v}"
        for k, v in state["borrower_profile"].items()
    )

    prompt = f"""You are a senior credit underwriter AI assistant at a regulated financial institution.

Analyse the following borrower information and produce a structured lending assessment.

━━━ BORROWER PROFILE ━━━
{borrower_text}

━━━ ML RISK ASSESSMENT ━━━
  • Default Probability   : {state['risk_score']:.2%}
  • Risk Classification   : {state['risk_label']}

━━━ KEY RISK DRIVERS (SHAP — positive values increase default risk) ━━━
{feature_text}

━━━ RETRIEVED POLICY CONTEXT ━━━
{policy_text}

━━━ INSTRUCTIONS ━━━
Respond with ONLY a valid JSON object (no markdown, no code fences).
Keys must be exactly as shown:

{{
  "profile_summary":    "2–3 sentence summary of the borrower's financial standing.",
  "risk_analysis":      "Explanation of the risk classification, referencing the key risk drivers.",
  "decision":           "APPROVE | CONDITIONAL APPROVE | DECLINE",
  "decision_rationale": "Justification for the decision, citing policy thresholds and risk factors.",
  "risk_mitigation":    "Concrete, actionable suggestions for the borrower to reduce risk.",
  "references":         "Policy documents or regulatory standards cited in this assessment.",
  "disclaimer":         "Legal and ethical disclaimer for this AI-generated assessment."
}}"""

    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        content  = response.content.strip()

        # Strip markdown fences if present
        content = re.sub(r"^```(?:json)?\s*", "", content, flags=re.MULTILINE)
        content = re.sub(r"\s*```\s*$",        "", content, flags=re.MULTILINE)

        report = json.loads(content)
    except Exception as exc:
        # Fallback to rule-based engine on LLM failure
        report = generate_lending_assessment(
            state["borrower_profile"],
            state["risk_score"],
            state["risk_label"],
            state["top_features"],
            state.get("policy_chunks")
        )

    return {**state, "report": report}


# ─── Template fallback ────────────────────────────────────────────────────────

# (Redundant _fallback_report removed, using src.explain_rules instead)


# ─── LangGraph agent factory ──────────────────────────────────────────────────

def build_agent(groq_api_key: str):
    """Compile the two-node LangGraph lending assessment agent."""
    from langgraph.graph import END, StateGraph
    from langchain_groq import ChatGroq

    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=groq_api_key,
        temperature=0.1,
        max_tokens=2048,
    )

    def _gen_report(state: LendingState) -> LendingState:
        return node_generate_report(state, llm)

    workflow = StateGraph(LendingState)
    workflow.add_node("retrieve_policies", node_retrieve_policies)
    workflow.add_node("generate_report",   _gen_report)
    workflow.set_entry_point("retrieve_policies")
    workflow.add_edge("retrieve_policies", "generate_report")
    workflow.add_edge("generate_report",   END)

    return workflow.compile()


# ─── Public API ───────────────────────────────────────────────────────────────

def run_lending_agent(
    borrower_profile: Dict[str, Any],
    risk_score:       float,
    risk_label:       str,
    top_features:     List[tuple],
    groq_api_key:     Optional[str] = None,
) -> Dict[str, str]:
    """Run the full agentic lending assessment pipeline.

    Parameters
    ----------
    borrower_profile :
        Raw borrower attribute dict (keys = field names, values = numbers).
    risk_score :
        Model probability of default in [0, 1].
    risk_label :
        Human-readable risk classification string.
    top_features :
        List of (feature_name, shap_value) tuples sorted by importance.
    groq_api_key :
        Optional Groq API key.  Falls back to ``GROQ_API_KEY`` env var then
        to the template-based report generator.

    Returns
    -------
    dict
        Structured lending assessment with keys: ``profile_summary``,
        ``risk_analysis``, ``decision``, ``decision_rationale``,
        ``risk_mitigation``, ``references``, ``disclaimer``.
    """
    # Priority: explicit arg → env var → Streamlit secrets
    api_key = groq_api_key or os.environ.get("GROQ_API_KEY")
    if not api_key:
        try:
            import streamlit as st
            api_key = st.secrets.get("GROQ_API_KEY", "")
        except Exception:
            pass

    initial: LendingState = {
        "borrower_profile": borrower_profile,
        "risk_score":       risk_score,
        "risk_label":       risk_label,
        "top_features":     top_features,
        "policy_chunks":    [],
        "report":           {},
    }

    if api_key:
        try:
            agent  = build_agent(api_key)
            result = agent.invoke(initial)
            return result["report"]
        except Exception:
            pass  # Fall through to template fallback

    # No key or LLM failure → rule-based assessment with RAG context
    state = node_retrieve_policies(initial)
    return generate_lending_assessment(
        state["borrower_profile"],
        state["risk_score"],
        state["risk_label"],
        state["top_features"],
        state.get("policy_chunks")
    )
