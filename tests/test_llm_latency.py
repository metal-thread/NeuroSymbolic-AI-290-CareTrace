import os
import time
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

def test_latency():
    load_dotenv()
    llm = ChatGoogleGenerativeAI(
        model="gemini-3-flash-preview",
        temperature=0.0,
        google_api_key=os.environ.get("GEMINI_API_KEY"),
        model_kwargs={
            "thinking": {"include_thoughts": False, "thinking_level": "minimal"},
            "tool_calling_method": "json_schema"
        }
    )
    
    msg = HumanMessage(content="My 6-year-old has a fever, threw up once, and looks really wiped out. Temp is 101.8. Extract symptoms and age.")
    
    print("Sending request to Gemini 3...")
    start = time.time()
    res = llm.invoke([msg])
    duration = time.time() - start
    print(f"Response received in {duration:.4f}s")
    print(f"Content: {res.content}")

if __name__ == "__main__":
    test_latency()
