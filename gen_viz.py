
# gen_viz.py

from __future__ import annotations
import os
import json
from typing import Dict, Any, List

# Import the actual Gemini SDK client
try:
    from google import genai
    HAS_GEMINI_SDK = True
except ImportError:
    HAS_GEMINI_SDK = False

# Environment variable key
GEMINI_API_KEY_KEY = "GEMINI_API_KEY"


def _static_fallback_viz(summary_stats: Dict[str, Any]) -> Dict:
    """Provides a safe, static Plotly specification on failure."""
    income = summary_stats.get('total_income', 0)
    expense = summary_stats.get('total_expense', 0)
    
    return {
        "type": "plotly_json",
        "description": "Fallback: Simple Income vs. Expense Donut Chart.",
        "spec": {
            "data": [
                {
                    "values": [income, expense],
                    "labels": ["Income", "Expense"],
                    "domain": {"x": [0, 1]},
                    "name": "I/E",
                    "hoverinfo": "label+percent+value",
                    "hole": 0.4,
                    "type": "pie",
                    "marker": {"colors": ["#0ea5e9", "#ff79b0"]}
                }
            ],
            "layout": {
                "title": {"text": "Total Financial Flow (Fallback)"},
                "height": 400,
                "template": "plotly_dark"
            }
        }
    }


def suggest_infographic_spec(summary_stats: Dict[str, Any], spikes: List[Dict]) -> Dict:
    """
    Calls Gemini to generate a Plotly figure specification based on financial data.
    
    Args:
        summary_stats: High-level financial metrics.
        spikes: List of spending spikes detected.
        
    Returns:
        Dict: A dictionary containing the visualization spec or a fallback.
    """
    if not HAS_GEMINI_SDK:
        return _static_fallback_viz(summary_stats)

    api_key = os.environ.get(GEMINI_API_KEY_KEY)
    if not api_key:
        return {"error": "Missing GEMINI_API_KEY", "spec": _static_fallback_viz(summary_stats)['spec']}

    try:
        client = genai.Client(api_key=api_key)
        
        data_summary = f"""
        Summary Statistics:
        - Total Income: {summary_stats.get('total_income', 0):.0f}
        - Total Expense: {summary_stats.get('total_expense', 0):.0f}
        - Net Savings: {summary_stats.get('net_savings', 0):.0f}
        - Top Expense Categories (Amount): {summary_stats.get('top_expenses', {})}
        - Top Spending Spikes (Last 30D): {spikes}
        """

        system_prompt = f"""
        You are a Data Visualization Specialist. Your task is to analyze the provided financial data summary
        and suggest the BEST single Plotly figure visualization (line, bar, pie, or scatter) that highlights 
        the most important financial insight, especially focusing on 'Net Savings' or 'Top Spending Spikes'.
        
        Respond ONLY with a single JSON object. DO NOT include any explanatory text outside the JSON.
        The JSON MUST have the following structure and MUST contain valid Plotly JSON specification in the 'spec' field.
        
        JSON Structure:
        {{
          "type": "plotly_json",
          "description": "A concise description of the chart and its insight (max 1 sentence).",
          "spec": {{ ... Plotly figure JSON here ... }}
        }}

        Constraints for the Plotly JSON 'spec':
        1. Use 'plotly_dark' template.
        2. Set a fixed height of 400.
        3. Use a clear, informative title.
        4. Focus on 'Top Expense Categories' or 'Spending Spikes' data.
        5. Use colors from the pink/blue theme (e.g., #ff79b0 for expense, #0ea5e9 for income/net).
        """

        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=[
                {"role": "user", "parts": [{"text": system_prompt + data_summary}]}
            ]
        )
        
        # Attempt to parse the JSON response
        json_string = response.text.strip().replace("```json", "").replace("```", "").strip()
        spec_dict = json.loads(json_string)
        
        # Simple validation
        if 'type' in spec_dict and spec_dict['type'] == 'plotly_json' and 'spec' in spec_dict:
            return spec_dict

    except Exception as e:
        print(f"Gemini/Plotly generation failed: {e}")

    # Fallback if API fails or response is not valid
    return _static_fallback_viz(summary_stats)