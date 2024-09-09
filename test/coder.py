import requests
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

from nlp_functions import extract_text_from_txt

API_URL = "https://ra8982at2iqrl8e3.eu-west-1.aws.endpoints.huggingface.cloud"
headers = {
    "Accept": "application/json",
    "Authorization": "Bearer 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",  # Replace with your actual token
    "Content-Type": "application/json"
}

# tokenizer = AutoTokenizer.from_pretrained("sambanovasystems/SambaCoder-nsql-llama-2-70b")
# model = AutoModelForCausalLM.from_pretrained("sambanovasystems/SambaCoder-nsql-llama-2-70b")

# tokenizer = AutoTokenizer.from_pretrained("defog/llama-3-sqlcoder-8b")
# model = AutoModelForCausalLM.from_pretrained("defog/llama-3-sqlcoder-8b")

# def query(payload):
#     # response = requests.post(API_URL, headers=headers, json=payload)
#     # return response.json()
#     inputs = tokenizer(payload, return_tensors="pt")
#     outputs = model.generate(inputs.input_ids, max_length=512, num_return_sequences=1)
#
#     # Decode the generated SQL query
#     return tokenizer.decode(outputs[0], skip_special_tokens=True)


def query_api(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()


def generate_sql_query(question, table_statements, instructions=""):
    prompt = f"""
    Generate a SQL query to answer this question: `{question}`
    {instructions}

    DDL statements:
    {table_statements}

    The following SQL query best answers the question `{question}`:
    ```sql
    """

    payload = {
        "inputs": prompt,
        "parameters": {"max_length": 8096, "num_return_sequences": 1, "temperature": 0.01}
    }

    response = query_api(payload)
    #response = query(prompt)
    print(response)
    return response[0]['generated_text']


# Example user question and table schemas
#

# instructions = "Use sessions,session_days,classrooms, addresses tables"
# user_question = "How many future sessions are in paris?"

instructions = "Use sessions, classrooms, addresses tables"
user_question = "Which classrooms are situated in zip code 06110"

# sql = extract_text_from_txt('')
# sql_query = generate_sql_query(user_question,  sql, instructions)
# print(sql_query)
