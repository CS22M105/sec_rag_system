"""
Simple OpenAI API test
"""
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Check if API key exists
api_key = os.getenv('OPENAI_API_KEY')
if not api_key:
    print("❌ ERROR: OPENAI_API_KEY not found in .env file")
    print("Please create a .env file with your API key:")
    print('OPENAI_API_KEY=sk-proj-your-key-here')
    exit(1)

print(f"✅ API key found: {api_key[:20]}...")

# Test import
try:
    from openai import OpenAI
    print("✅ OpenAI package imported successfully")
except ImportError as e:
    print(f"❌ ERROR importing OpenAI: {e}")
    print("Try: pip install --upgrade openai")
    exit(1)

# Test API connection
try:
    client = OpenAI(api_key=api_key)
    print("✅ OpenAI client created successfully")
    
    # Make a simple API call
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Say 'API works!'"}],
        max_tokens=10
    )
    
    answer = response.choices[0].message.content
    print(f"✅ API call successful!")
    print(f"   Response: {answer}")
    print(f"   Model used: {response.model}")
    
except Exception as e:
    print(f"❌ ERROR calling API: {e}")
    print("\nTroubleshooting:")
    print("1. Check your API key is correct")
    print("2. Verify you have credits: https://platform.openai.com/account/billing")
    print("3. Try: pip install --upgrade openai httpx")
    exit(1)

print("\n🎉 All checks passed! Your setup is ready.")