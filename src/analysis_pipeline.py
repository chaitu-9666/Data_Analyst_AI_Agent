
"""Connect natural-language query planning with safe execution."""

from src.query_planner import build_query_plan
from src.execution_engine import execute_plan_safely


def run_analysis(query, df):
    """
    Build a query plan and execute it when all required parameters
    are available.

    Returns the plan and a structured execution response.
    """

    try:
        # Step 1: Convert the natural-language query into a plan.
        plan = build_query_plan(query, df)

        tool_plan = plan.get("tool", {})
        tool_status = tool_plan.get("status")

        # Step 2: Do not execute incomplete or unsupported plans.
        if tool_status != "selected":
            return {
                "status": tool_status or "error",
                "plan": plan,
                "execution": None,
                "message": (
                    "The query could not be executed automatically. "
                    "The selected tool may be unsupported or may need "
                    "additional parameters."
                ),
            }

        # Step 3: Execute the approved plan safely.
        execution = execute_plan_safely(df, plan)

        return {
            "status": execution["status"],
            "plan": plan,
            "execution": execution,
            "message": execution.get("message"),
        }

    except (TypeError, ValueError) as error:
        return {
            "status": "error",
            "plan": None,
            "execution": None,
            "message": str(error),
        }

    except Exception:
        return {
            "status": "error",
            "plan": None,
            "execution": None,
            "message": (
                "An unexpected error occurred while processing "
                "your question."
            ),
        }
