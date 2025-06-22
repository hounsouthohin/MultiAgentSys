from smolagents import LiteLLMModel

model = LiteLLMModel(
    model="ollama/llama3.2",
    model_id="ollama/llama3.2",
    base_url="http://localhost:11434"
)

messages = [
  {"role": "user", "content": [{"type": "text", "text": "Hello, how are you?"}]}
]

response = model(messages)
print(response.content)


###Question :  why thid model just answer the question, and not purchase the conversation?


