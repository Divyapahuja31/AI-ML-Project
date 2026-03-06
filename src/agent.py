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
        report = _fallback_report(state, error=str(exc))

    return {**state, "report": report}


# ─── Template fallback ────────────────────────────────────────────────────────

def _fallback_report(state: LendingState, error: str = "") -> Dict[str, str]:
    """Generate a deterministic, template-based report when the LLM is unavailable."""
    score   = state["risk_score"]
    label   = state["risk_label"]
    profile = state["borrower_profile"]

    if score >= 0.50:
        decision   = "DECLINE"
        rationale  = (
            f"The ML model assigns a {score:.1%} default probability, exceeding the "
            "institution's 50% maximum threshold for standard lending. The elevated "
            "delinquency history and high credit utilization further support this decision."
        )
        mitigation = (
            "Improve revolving utilization below 30% over the next 12 months. "
            "Demonstrate 12 consecutive months of on-time payments. "
            "Consider credit counselling to reduce the debt-to-income ratio below 43%."
        )
    elif score >= 0.30:
        decision   = "CONDITIONAL APPROVE"
        rationale  = (
            f"The default probability of {score:.1%} is elevated (30–50% range). "
            "Conditional approval may be extended subject to compensating factors such as "
            "co-signer, collateral, or reduced loan amount (≤70% of requested)."
        )
        mitigation = (
            "Require a creditworthy co-signer or appropriate collateral. "
            "Cap credit limit at 70% of the requested amount. "
            "Enrol in monthly auto-debit repayment and conduct quarterly account reviews."
        )
    else:
        decision   = "APPROVE"
        rationale  = (
            f"The default probability of {score:.1%} is within the acceptable low-risk "
            "range (<30%). Repayment history and financial profile support standard approval."
        )
        mitigation = (
            "Standard annual account review applies. "
            "Monitor for any increase in revolving utilization above 60%."
        )

    top_factors = ", ".join(f[0].replace("_", " ") for f in state.get("top_features", [])[:3])

    return {
        "profile_summary": (
            f"Borrower aged {profile.get('age', 'N/A')} with monthly income of "
            f"${profile.get('monthly_inc', 0):,.0f} and {profile.get('dependents', 0)} "
            f"dependent(s). Revolving utilization: {profile.get('rev_util', 0):.2f}. "
            f"Debt ratio: {profile.get('debt_ratio', 0):.2f}."
        ),
        "risk_analysis": (
            f"The ML model (Random Forest, ROC-AUC optimised) classifies this applicant "
            f"as **{label}** with a {score:.2%} probability of delinquency within 2 years. "
            f"Primary risk drivers (by SHAP magnitude): {top_factors}."
        ),
        "decision":           decision,
        "decision_rationale": rationale,
        "risk_mitigation":    mitigation,
        "references": (
            "Internal Credit Assessment Standards v2.3; "
            "Risk Threshold & Decision Policy; "
            "Fair Lending Compliance Policy; "
            "Basel III Credit Risk Framework (IRB Approach)."
        ),
        "disclaimer": (
            "This assessment is generated by an AI system for informational purposes only. "
            "Final lending decisions must be reviewed and authorised by a licensed credit "
            "professional. All decisions must comply with applicable consumer protection "
            "and fair lending laws, including ECOA, the Fair Housing Act, and Regulation B. "
            "This report does not constitute a binding credit decision."
        ),
    }


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

    # No key or LLM failure → template-based with RAG context
    state = node_retrieve_policies(initial)
    return _fallback_report(state)
