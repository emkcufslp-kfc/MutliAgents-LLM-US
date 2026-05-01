import os
from dotenv import load_dotenv

load_dotenv()

def test_fred():
    fred_key = os.environ.get("FRED_API_KEY")
    if not fred_key or fred_key == "your_fred_api_key_here":
        print("[FAIL] FRED_API_KEY is missing or unchanged.")
        return
        
    try:
        from fredapi import Fred
        fred = Fred(api_key=fred_key)
        val = fred.get_series('UNRATE').iloc[-1]
        print(f"[SUCCESS] FRED API connected successfully! (Latest UNRATE: {val}%)")
    except Exception as e:
        print(f"[FAIL] FRED API Failed: {e}")

def test_grok():
    xai_key = os.environ.get("XAI_API_KEY")
    if not xai_key or xai_key == "your_xai_api_key_here":
        print("[FAIL] XAI_API_KEY is missing or unchanged.")
        return
        
    try:
        from langchain_openai import ChatOpenAI
        from langchain_core.messages import HumanMessage
        llm = ChatOpenAI(
            model="grok-3-mini",
            api_key=xai_key,
            base_url="https://api.x.ai/v1",
            temperature=0.1,
        )
        resp = llm.invoke([HumanMessage(content="Hello, respond with 'OK'")])
        print(f"[SUCCESS] xAI Grok API connected successfully! (Response: {resp.content})")
    except Exception as e:
        print(f"[FAIL] xAI Grok API Failed: {e}")

if __name__ == "__main__":
    print("\n--- Testing API Connections ---")
    test_fred()
    test_grok()
    print("-------------------------------\n")
