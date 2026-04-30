import os
import yaml
from pathlib import Path
import logging
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# To prevent crashing if the user hasn't installed specific LLM packages yet
try:
    from langchain_community.chat_models import ChatOllama
except ImportError:
    ChatOllama = None

try:
    from langchain_google_genai import ChatGoogleGenerativeAI
except ImportError:
    ChatGoogleGenerativeAI = None

class LLMFactory:
    """
    Reads config/settings.yaml and initializes the appropriate LangChain models
    for Tier 1 (Fast/Local) and Tier 2 (Deep Reasoning/Cloud).
    """
    
    def __init__(self):
        self.config = self._load_config()
        
    def _load_config(self) -> dict:
        config_path = Path(__file__).parent.parent.parent / "config" / "settings.yaml"
        if not config_path.exists():
            logger.warning(f"Config not found at {config_path}. Using defaults.")
            return {}
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
            
    def get_tier1_llm(self):
        """Tier 1: Typically Ollama for free, fast data extraction."""
        provider = self.config.get("llm", {}).get("tier1_provider", "ollama")
        model_name = self.config.get("llm", {}).get("tier1_model", "llama3")
        
        logger.info(f"Initializing Tier 1 LLM: {provider} / {model_name}")
        
        if provider == "ollama" and ChatOllama:
            # Requires Ollama running locally on port 11434
            return ChatOllama(model=model_name, temperature=0.1)
        elif provider == "gemini" and ChatGoogleGenerativeAI:
            api_key = os.environ.get("GOOGLE_API_KEY")
            if not api_key:
                logger.error("GOOGLE_API_KEY not found.")
                return None
            return ChatGoogleGenerativeAI(model=model_name, temperature=0.1, convert_system_message_to_human=True)
        else:
            logger.warning(f"Tier 1 provider '{provider}' not available or package missing. Returning None.")
            return None

    def get_tier2_llm(self):
        """Tier 2: Typically Gemini/Claude/GPT-4 for complex debate and Portfolio Manager decisions."""
        provider = self.config.get("llm", {}).get("tier2_provider", "gemini")
        model_name = self.config.get("llm", {}).get("tier2_model", "gemini-1.5-pro-latest")
        
        logger.info(f"Initializing Tier 2 LLM: {provider} / {model_name}")
        
        if provider == "gemini" and ChatGoogleGenerativeAI:
            api_key = os.environ.get("GOOGLE_API_KEY")
            if not api_key:
                logger.error("GOOGLE_API_KEY not found in environment variables.")
                return None
            return ChatGoogleGenerativeAI(model=model_name, temperature=0.2, convert_system_message_to_human=True)
        else:
            logger.warning(f"Tier 2 provider '{provider}' not available or package missing. Returning None.")
            return None
