import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Snowflake Cortex LLM

## Configuration
Set environment variables for snowflake connection. 


## Sample Usage
```python
from litellm import completion
import os

os.environ['SNOWFLAKE_ACCOUNT']=""
os.environ['SNOWFLAKE_SERVICE_USER']=""
os.environ['SNOWFLAKE_AUTHMETHOD']=""
os.environ['SNOWFLAKE_PASSWORD_KEY']=""
os.environ['SNOWFLAKE_URL']=""
os.environ['PERPLEXITYAI_API_KEY'] = ""
response = completion(
    model="snowflake_cortex/mistral-large2",
    messages=[{"role": "user", "content": "Hi"}]
)
print(response)
```

## Supported methods
For the moment only is supported syncrhonos complete method 
