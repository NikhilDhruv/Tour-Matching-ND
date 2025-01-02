from openai import OpenAI
import os
from dotenv import load_dotenv
from celery_app import api_call  # Import the function from celery_app.py

load_dotenv()  # Load environment variables

# Test input for the API call
test_input = "Sample input"

# Call the function and print the embedding
embedding = api_call(test_input)
print("Embedding:", embedding)

