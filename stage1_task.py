import os
from dotenv import load_dotenv
import time
import random
from typing import List, Any
try:
    from huggingface_hub import InferenceClient
except ImportError:
    InferenceClient = None


load_dotenv()
hf_token = os.getenv("HF_TOKEN")
repo_id = "Qwen/Qwen2.5-72B-Instruct"
client = None

if InferenceClient is None:
    print("Error: huggingface_hub not installed or failed to import.")
else:
    try:
        client = InferenceClient(repo_id, token=hf_token)
    except Exception as e:
        print("Error creating InferenceClient")
        client = None

class Synapse:
    """
    Operator to pass data context forward
    """
    def __init__(self, prompt_template: str, name: str):
        self.template = prompt_template
        self.name = name

    @staticmethod
    def _parse_hf_response(resp: any) -> str:
        if isinstance(resp, str):
            return resp.strip()

        try:
            content = getattr(resp.choices[0].message, "content", None)

            if content:
                return str(content).strip()
        except Exception:
            pass

        try:
            if isinstance(resp, dict):
                return str(resp["choices"][0]["message"]["content"]).strip()
        except Exception:
            pass

        try:
            return str(resp).strip()
        except Exception:
            return ""

    def _fire_synapse(self, context_str: str, original_query: str) -> str:
        if client is None:
            return "Error: InferenceClient not initialized. HF client is not available."

        print(f" Processing {self.name}...")
        prompt_content = self.template.replace("{{input}}", context_str)
        prompt_content = prompt_content.replace("{{previous_output}}", context_str)
        prompt_content = prompt_content.replace("{{original_query}}", original_query)

        system_instruction = "You are an intelligent customer support system for a bank that converses with customers and helps solve their problems.. Output ONLY the requested value. Do not converse or make extra commentary."

        messages = [
                    {"role": "system", "content": system_instruction},
                    {"role": "user", "content": prompt_content}
        ]

        try:
            response = client.chat_completion(
                model=repo_id,
                messages=messages,
                max_tokens=200,
                temperature=0.1,
                stream=False
            )

            output_text = self._parse_hf_response(response)
            output_text = output_text.replace("[/ASSIST]", "").replace("[ASSIST]", "").strip()
            if "\n" in output_text and "step" not in self.name.lower():
                 for line in output_text.splitlines():
                    line = line.strip()
                    if line:
                        output_text = line
                        break

            return output_text
        except Exception as e:
            return f"Error in _fire_synapse: {type(e)}"

    def __rshift__(self, next_step):
        """
        Allows chaining: Step1 >> Step2 >> Step3
        """
        return NeuralChain(self, next_step)

class NeuralChain:
    """
    Holds sequence of synapses and executes them like a domino effect
    """
    def __init__(self, first, second):
        self.steps = []

        if isinstance(first, NeuralChain):
            self.steps.extend(first.steps)
        else:
            self.steps.append(first)
        self.steps.append(second)

    def __rshift__(self, next_step):
        self.steps.append(next_step)
        return self

    def execute(self, initial_input: str) -> list:
        results = []
        current_context = initial_input
        original_query = initial_input

        print(f"\n SYSTEM IGNITION FOR: '{initial_input}'")
        print("="*60)

        for i, synapse in enumerate(self.steps):

            # Added this to show which step is running clearly
            print(f"\n🔹 STEP {i+1}: {synapse.name}")

            # Special Handling for Step 5
            if i == 4 and len(results) >= 4:
                current_context = synapse.template
                current_context = current_context.replace("{{category}}", results[2])
                current_context = current_context.replace("{{missing_info}}", results[3])

            # Run the step
            output = synapse._fire_synapse(current_context, original_query)

            # Added this to print the result cleanly immediately
            print(f"   └── Result: {output}")

            results.append(output)
            current_context = output

        print("\n" + "="*60)
        return results

# Internet intent
step_1 = Synapse(
    prompt_template="Analyze raw text: '{{input}}'. Output format: 'User intends to [action] regarding [subject]. Be specific, factual, and customer centered'",
    name="1_INTERPRET_INTENT"
)

# Map categories
step_2 = Synapse(
    prompt_template="Intent: '{{previous_output}}'. Suggest 3 categories from [Account Opening, Billing Issue, Account Access, Transaction Inquiry, Card Services, Account Statement, Loan Inquiry, General Information].",
    name="2_MAP_CATEGORIES"
)

# Choose category
step_3 = Synapse(
    prompt_template="Candidates: [{{previous_output}}]. Pick the SINGLE best category.",
    name="3_CHOOSE_CATEGORY"
)

# Extract details
step_4 = Synapse(
    prompt_template=(
        "Category: '{{previous_output}}'."
        "Original customer message: '{{original_query}}'."
        "From customer message, identify and extract REQUIRED information and PROVIDED information that could be a part of the missing information in the next step."
        "Only ask for missing information not present in customer information. Do NOT repeat information already provided as missing information."
        "Create a new {{missing_info}} and pass it to the next step."
    ),
    name="4_EXTRACT_DETAILS"
)

# Generate response
step_5 = Synapse(
    prompt_template="Context: {{category}}. Missing: {{missing_info}}. Draft polite response. Explain {{query}}. Reassure user. Politely ask for missing info clearly stating them in a bracket. Do not use placeholders like {{customer_name}}. Maximum of 3-5 sentences long.",
    name="5_GENERATE_RESPONSE"
)

def run_prompt_chain(query: str) -> List[str]:
    logic_flow = step_1 >> step_2 >> step_3 >> step_4 >> step_5
    chain_result = logic_flow.execute(query)

    return chain_result[-1]

def query(customer_query):
    query = customer_query
    output = run_prompt_chain(query)
    return output

customer_query = input(f"Enter your query: ")
print(query(customer_query))
