import logging
from typing import Any, Dict, List, Optional, Protocol, Union

logger = logging.getLogger(__name__)


class RewardScorer(Protocol):
    """
    Protocol for calculating a reward based on the input state (before action)
    and the result state (after action).
    """

    def score(self, params: Dict[str, Any]) -> float:
        """
        Calculate reward.
        params usually contains: input, result, error, trace, etc.
        """
        ...


class ErrorScorer:
    """
    Assigns a penalty if specific error keys exist in the result state,
    or if an exception object is passed.
    """

    def __init__(
        self,
        error_keys: List[str] = ["error", "exception"],
        penalty: float = -1.0,
        success_reward: float = 1.0,
    ):
        self.error_keys = error_keys
        self.penalty = penalty
        self.success_reward = success_reward

    def score(self, result: Any) -> float:
        """
        Checks result state (dict) for error keys.
        """
        if isinstance(result, dict):
            for key in self.error_keys:
                if result.get(key):
                    return self.penalty
        # If result is an Exception object itself (unlikely in pure state dicts but possible in frameworks)
        if isinstance(result, Exception):
            return self.penalty

        return self.success_reward


class LLMScorer:
    """
    Uses an LLM to judge the quality of the interaction.
    Requires an llm callable (e.g., langchain Runnable or function).
    """

    def __init__(self, llm_callable: Any, prompt_template: str):
        """
        Args:
            llm_callable: Function that takes string prompt and returns string response.
            prompt_template: String with {query} and {response} placeholders.
                             Should ask LLM to output a single float number.
        """
        self.llm = llm_callable
        self.prompt = prompt_template

    def score(self, query: str, response: str) -> float:
        """
        Constructs prompt, calls LLM, parses float.
        """
        formatted_prompt = self.prompt.format(query=query, response=response)
        try:
            # support langchain Runnable or simple func
            if hasattr(self.llm, "invoke"):
                result = self.llm.invoke(formatted_prompt)
                # If result is an AIMessage, get content
                if hasattr(result, "content"):
                    text = result.content
                else:
                    text = str(result)
            else:
                text = str(self.llm(formatted_prompt))

            # Naive parsing: find the first number
            import re

            match = re.search(r"[-+]?\d*\.\d+|\d+", text)
            if match:
                return float(match.group())
            return 0.0
        except Exception as e:
            logger.error(f"LLMScorer failed to parse response: {e}")
            return 0.0
