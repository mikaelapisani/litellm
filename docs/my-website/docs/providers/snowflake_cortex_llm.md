import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Snowflake Cortex LLM

## Configuration
| Variable         | Description    |
| ---------------- | -------------- |
| SNOWFLAKE_ACCOUNT  | Snowflake account name |
| SNOWFLAKE_SERVICE_USER | Username to connect to Snowflake     |
|SNOWFLAKE_URL|is the url for the Snowflake LLM Service. For example : http://{account-name}/snowflakecomputing.com/api/v2/cortex/inference:complete |
| SNOWFLAKE_AUTHMETHOD    | Auth method can be 'oauth' (when running inside snowflake ecosystem) or 'jwt' (when running otside snowflake, e.g locally)  |
|SNOWFLAKE_PASSWORD_KEY| Only required for JWT authentication. . See guidelines below on how to create the key |

## Sample Usage
```python
from litellm import completion
import os
os.environ['SNOWFLAKE_ACCOUNT']= ""
os.environ['SNOWFLAKE_SERVICE_USER']= ""
os.environ['SNOWFLAKE_URL']= ""
os.environ['SNOWFLAKE_AUTHMETHOD']= ""
os.environ['SNOWFLAKE_PASSWORD_KEY']= ""
response = completion(
    model="snowflake_cortex/mistral-large2",
    messages=[{"role": "user", "content": "Hi"}]
)
print(response)
```

## Supported methods
Currently, only the synchronous `completion` method is available.

Tool calling isn't supported by the provider yet, but we're actively working on adding this functionality in an upcoming release.

## Using Cortex LLM service through the API

Snowflake Cortex is exposed as an HTTP API by default. You can access it by setting up key-pair authentication and providing a valid token.

This method is recommended due to its superior reliability, including:
	1.	Improved Handling of Special Characters: Provides better support for special characters, ensuring robust performance in various scenarios.
	2.	Reduced Sensitivity to SQL Formatting: More forgiving about SQL formatting, ensuring a smoother integration experience compared to other methods.

See more information at the following link [Cortex LLM REST API](https://docs.snowflake.com/en/user-guide/snowflake-cortex/cortex-llm-rest-api)

## Grant Permissions in Snowflake
Ensure the service has the necessary role access by running the following commands:

```sql
    GRANT DATABASE ROLE snowflake.cortex_user TO ROLE MY_ROLE;
    GRANT ROLE MY_ROLE TO USER MY_USER;
```

## Guideline to create key pair auth
Based on the following tutorials:   
- [Key Pair Authentication in Snowflake](https://docs.snowflake.com/en/user-guide/key-pair-auth)   
- [Snowflake REST API Authentication](https://docs.snowflake.com/en/developer-guide/snowflake-rest-api/authentication)   

1. Generate a Private Key  
```bash
    openssl genrsa 2048 | openssl pkcs8 -topk8 -v2 des3 -inform PEM -out rsa_key.p8
```
2. Generate the Public Key  
```bash
    openssl rsa -in rsa_key.p8 -pubout -out rsa_key.pub
```
3. Alter the User Table with the Public Key  
```sql
    ALTER USER example_user SET RSA_PUBLIC_KEY='MIIBIjANBgkqh...';
```
4. Obtain the SHA256 Value of the Public Key  
```bash
    openssl rsa -pubin -in rsa_key.pub -outform DER | openssl dgst -sha256 -binary | openssl enc -base64
```
## Supported Models
For a list of available models and their regions, refer to the [Model Availability Documentation](https://docs.snowflake.com/en/user-guide/snowflake-cortex/cortex-llm-rest-api#model-availability)
