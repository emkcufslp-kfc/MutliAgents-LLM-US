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

def test_gemini():
    gemini_key = os.environ.get("GOOGLE_API_KEY")
    if not gemini_key or gemini_key == "your_gemini_api_key_here":
        print("[FAIL] GOOGLE_API_KEY is missing or unchanged.")
        return
        
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
        from langchain_core.messages import HumanMessage
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=gemini_key, convert_system_message_to_human=True)
        resp = llm.invoke([HumanMessage(content="Hello, respond with 'OK'")])
        print(f"[SUCCESS] Google Gemini API connected successfully! (Response: {resp.content})")
    except Exception as e:
        print(f"[FAIL] Google Gemini API Failed: {e}")

if __name__ == "__main__":
    print("\n--- Testing API Connections ---")
    test_fred()
    test_gemini()
    print("-------------------------------\n")
