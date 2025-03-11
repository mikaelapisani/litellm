from litellm import completion

response = completion(
    model="snowflake_cortex/mistral-large2",
    messages=[{"role": "user", "content": "Hi"}]
)
print(response)




