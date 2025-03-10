import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Snowflake Cortex LLM

## Configuration

## Sample Usage
```python
from litellm import completion
import os

os.environ['PERPLEXITYAI_API_KEY'] = ""
response = completion(
    model="perplexity/mistral-7b-instruct", 
    messages=messages
)
print(response)
```

## Supported methods
For the moment only is supported syncrhonos complete method 
