import os
from google import genai
from dotenv import load_dotenv

def test_gemini_connection():
    # Load environment variables
    load_dotenv()
    
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("[FAIL] GEMINI_API_KEY is not set in the environment or .env file.")
        return False
        
    print(f"Loaded GEMINI_API_KEY: {api_key[:6]}...{api_key[-4:] if len(api_key) > 10 else ''}")
    
    try:
        # Initialize Google GenAI client
        print("Initializing GenAI Client...")
        client = genai.Client(api_key=api_key)
        
        # Test generation with a lightweight model
        print("Sending test request to 'gemini-2.5-flash'...")
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents='Respond with only the word "SUCCESS" if you receive this message.'
        )
        
        text = response.text.strip()
        print(f"Received Response: '{text}'")
        if "SUCCESS" in text.upper():
            print("[OK] Gemini API key works and connection is successful!")
            return True
        else:
            print("[WARNING] Received unexpected response, but the API key/connection is working.")
            return True
            
    except Exception as e:
        print(f"[FAIL] Gemini API connection failed: {e}")
        return False

if __name__ == "__main__":
    test_gemini_connection()
