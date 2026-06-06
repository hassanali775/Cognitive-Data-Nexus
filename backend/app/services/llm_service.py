import ollama
from typing import List, Dict, Any

class LocalLLMService:
    """
    Air-gapped Local Inference Service.
    Routes fused context matrices straight into a localized Ollama engine
    to generate grounded, hallucination-free enterprise responses.
    """
    def __init__(self, model_name: str = "llama3"):
        self.model_name = model_name

    def generate_answer(self, query: str, vector_context: List[str], graph_context: List[str]) -> str:
        """Fuses structural data frames and streams localized generation context boundaries."""
        
        # Format contexts neatly into clean string blocks
        vector_str = "\n".join([f"- {ctx}" for ctx in vector_context])
        graph_str = "\n".join([f"- {edge}" for edge in graph_context])
        
        # System instructions enforcing absolute data grounding
        system_prompt = (
            "You are the advanced analytical reasoning engine for Cognitive-Data-Nexus.\n"
            "Your objective is to answer the user query using ONLY the provided multi-dimensional context blocks.\n"
            "If the contexts do not contain the answer, state clearly that the localized data bounds are insufficient.\n"
            "Do not under any circumstances hallucinate outside facts."
        )
        
        user_prompt = f"""
[OBJECTIVE QUERY]
{query}

[RETRIEVED GEOMETRIC CONTEXT (Vector Space)]
{vector_str if vector_str else "No geometric context items returned."}

[RETRIEVED TOPOLOGICAL CONTEXT (Knowledge Graph)]
{graph_str if graph_str else "No topological relationship paths returned."}

[GENERATED ANSWER]
"""

        try:
            response = ollama.generate(
                model=self.model_name,
                system=system_prompt,
                prompt=user_prompt,
                options={
                    "temperature": 0.0,  # Force deterministic accuracy over creativity
                    "top_p": 0.1
                }
            )
            return response.get("response", "Error: Empty stream returned from local engine.")
        except Exception as e:
            return f"Inference Execution Crash: Failed to query local Ollama instance. Details: {str(e)}"