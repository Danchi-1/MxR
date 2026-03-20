import os
import json
from dotenv import load_dotenv
from openai import OpenAI

# optionally import Gemini's SDK if available
try:
    import google.generativeai as genai
except ImportError:
    genai = None

load_dotenv()

# choose a client factory based on environment variables
MODEL = os.getenv("LLM_MODEL_NAME", "openai/gpt-4o-mini")

# helper for building chat completions regardless of provider

def _create_chat_completion(messages, tools):
    """Dispatch to OpenAI/OpenRouter or Gemini based on env.

    - If GEMINI_API_KEY is set and the Gemini SDK is installed, use Gemini.
    - Otherwise default to OpenAI via the OpenRouter base URL.
    """
    if os.getenv("GEMINI_API_KEY") and genai is not None:
        # configure Gemini client each call (idempotent)
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        # the generativeai library uses `.chat.completions.create`
        return genai.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=tools,
        )

    # if we reach here, either gemini isn't requested/available or no key was provided
    openrouter_key = os.getenv("OPENROUTER_API_KEY")
    if not openrouter_key:
        raise EnvironmentError(
            "No API key provided. Set GEMINI_API_KEY to use Gemini or OPENROUTER_API_KEY to use OpenAI/OpenRouter."
        )

    client = OpenAI(
        api_key=openrouter_key,
        base_url="https://openrouter.ai/api/v1",
    )
    return client.chat.completions.create(
        model=MODEL,
        messages=messages,
        tools=tools,
    )



# ---------- TOOL DATA ----------

FLIGHTS = {
    ("Lagos", "Nairobi"): {
        "duration_hours": 5,
        "price_usd": 450
    }
}

HOTELS = {
    "Nairobi": 120
}

RATES = {
    "NGN": 1500,
    "EUR": 0.92,
    "GBP": 0.78,
    "USD": 1
}


# ---------- TOOL FUNCTIONS ----------

def get_flight_schedule(from_city: str, to_city: str):

    data = FLIGHTS.get((from_city, to_city))

    if not data:
        return {"error": "route not available"}

    return {
        "from": from_city,
        "to": to_city,
        "duration_hours": data["duration_hours"],
        "price_usd": data["price_usd"]
    }


def get_hotel_schedule(city: str):

    price = HOTELS.get(city)

    if not price:
        return {"error": "city not available"}

    return {
        "city": city,
        "price_per_night_usd": price
    }

def convert_currency(amount_usd: float, currency: str):

    rate = RATES.get(currency, 1)

    return {
        "amount_usd": amount_usd,
        "currency": currency,
        "converted_amount": amount_usd * rate
    }

# ---------- TOOL SCHEMA ----------

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_flight_schedule",
            "description": "Return flight duration and cost",
            "parameters": {
                "type": "object",
                "properties": {
                    "from_city": {"type": "string"},
                    "to_city": {"type": "string"}
                },
                "required": ["from_city", "to_city"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_hotel_schedule",
            "description": "Return hotel price per night",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string"}
                },
                "required": ["city"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "convert_currency",
            "description": "Convert USD to another currency",
            "parameters": {
                "type": "object",
                "properties": {
                    "amount_usd": {"type": "number"},
                    "currency": {"type": "string"}
                },
                "required": ["amount_usd", "currency"]
            }
        }
    }
]


# ---------- MAIN LOOP ----------

def main():

    messages = [
        {
            "role": "user",
            "content": "I'm taking a flight from Lagos to Nairobi for a conference. I would like to know the total flight time back and forth, and the total cost of logistics for this conference if I'm staying for three days."
        }
    ]

    # dispatch to the appropriate provider
    response = _create_chat_completion(messages, tools)

    message = response.choices[0].message

    while message.tool_calls:

        messages.append(message)

        for call in message.tool_calls:

            name = call.function.name
            args = json.loads(call.function.arguments)

            if name == "get_flight_schedule":
                result = get_flight_schedule(**args)

            elif name == "get_hotel_schedule":
                result = get_hotel_schedule(**args)

            elif name == "convert_currency":
                result = convert_currency(**args)

            else:
                result = {"error": "unknown tool"}

            messages.append({
                "role": "tool",
                "tool_call_id": call.id,
                "content": json.dumps(result)
            })

        response = _create_chat_completion(messages, tools)

        message = response.choices[0].message

    print(message.content)

if __name__ == "__main__":
    main()