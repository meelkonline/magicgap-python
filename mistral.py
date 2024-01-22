from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage

model = "mistral-tiny"
client = MistralClient(api_key="AnnCHxwRjdj38WrvDAcz3u23dTBF6VO0")

messages = [
    ChatMessage(role="user", content="What is the best French cheese?")
]

# No streaming
chat_response = client.chat(
    model=model,
    messages=messages,
)

# With streaming
for chunk in client.chat_stream(model=model, messages=messages):
    print(chunk)