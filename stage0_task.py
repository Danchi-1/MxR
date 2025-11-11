import os
from dotenv import load_dotenv
import requests
from mistralai import Mistral

# load secrets from environment
load_dotenv()
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
client = Mistral(MISTRAL_API_KEY)

# define improper words
banned_keywords = ['kill', 'hack', 'bomb', 'terror', 'murder']
prompt = input("Prompt Here: ")

# function to check if prompt is safe
def safe_prompt(prompt: str) -> bool:
    low_text_prompt = prompt.lower()
    return not any(word in low_text_prompt for word in banned_keywords)


def safe_output(text: str) -> str:
    safe_text = text
    for word in BANNED_KEYWORDS:
        safe_text = safe_text.replace(word, "[REDACTED]").replace(word.capitalize(), "[REDACTED]")
    return safe_text

def reply_prompt(prompt):
    if not safe_prompt(prompt):
        return "User prompt includes prohibited language"
    messages = [
        {"role": "system", "content": "You are a helpful assistant. You should explain any question asked very well while maintaining minimum words, never use markdown unless when explicitly asked to or required. You should reply prompts sent to you if they don't contain sensitive informations like kill, bomb, hack"},
        {"role": "user", "content": prompt}
    ]
    response = client.chat.complete(
        model="mistral-large-latest",
        messages=messages
    )
    output = response.choices[0].message.content
    return safe_output(output)
reply_prompt(prompt)
