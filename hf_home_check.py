import os
from huggingface_hub import HfFolder

# 1. Check Cache Path
print(f"HF_HOME is set to: {os.getenv('HF_HOME')}")

# 2. Check Login Status
token = HfFolder.get_token()
if token:
    print("✅ Logic Status: Successfully logged in!")
    print(f"Token: {token[:5]}...") # Prints first 5 chars only
else:
    print("❌ Not logged in.")