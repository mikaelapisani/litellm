import os 
from litellm import completion


response = completion(
    model="snowflake_cortex/snowflake-arctic",
    messages=[{"role": "user", "content": "Hi"}]
)
print(response)

