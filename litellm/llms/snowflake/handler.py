# Author: Sean Iannuzzi
# Created: January 13, 2024

import json
import time
from datetime import timedelta
from typing import Any, AsyncIterator, Dict, Iterator, List, Optional, Union

import litellm
import requests
from litellm import LiteLLMLoggingObj
from litellm.llms.base import BaseLLM
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.types.utils import GenericStreamingChunk, ModelResponse
from litellm.llms.snowflake.jwt_generator import JWTGenerator
from litellm.llms.snowflake.utils import generate_unique_id
)


class SnowflakeCortexInferenceService(BaseLLM):
    """
    SnowflakeCortexInferenceService: A custom implementation of a Snowflake Cortext Inference Service
    designed to integrate with the Snowflake API and be registered as a provider
    within LiteLLM and Crew AI agents.

    This class extends the base `CustomLLM` class to facilitate the integration with the Snowflake Cortext Inference Service API,
    allowing access to supported LLMs in Snowflake Cortex.
    See more info at https://docs.snowflake.com/en/user-guide/snowflake-cortex/llm-functions

    Attributes:
        model (str): The model name or identifier to be used for generating responses. See the documentation and select which one to use https://docs.snowflake.com/en/user-guide/snowflake-cortex/llm-functions#choosing-a-model
        snowflake_account (str): The Snowflake account name for authentication and data access.
        snowflake_service_user (str): The Snowflake service user name for API access.
        snowflake_authmethod: (Optional[str]): Sets auth method, can be 'oauth' or 'jwt'. By default is 'oauth'
        snowflake_token_path: Optional[str]: By default '/snowflake/session/token' , the route where the token is located in Snowpark Containers
        privatekey_password (Optional[str]): Private password used for key encription for generating jwt token.
        timeout (Optional[Union[float, int]]): The maximum time to wait for the model's response.
        temperature (Optional[float]): Adjusts the creativity/randomness of the model's outputs.
        top_p (Optional[float]): Applies nucleus sampling to limit token choices based on cumulative probability.
        n (Optional[int]): The number of completions to generate for each request.
        stop (Optional[Union[str, List[str]]]): Defines one or more stop sequences to halt the model's output.
        max_completion_tokens (Optional[int]): Maximum token count for the completion response.
        base_url (Optional[str]): The base URL for the Snowflake's API endpoint.
        api_key (Optional[str]): The API key required for authentication (For example -----BEGIN PRIVATE KEY----- .. -----END PRIVATE KEY-----).
        format_messages_callback (Optional[Any]): Is a function to be called before post request in order to preprocess history messages if needed
        **kwargs,


    Methods:
        __init__(self, **kwargs): Initializes the custom LLM with the provided configuration options.
        call(self, *args, **kwargs) -> litellm.ModelResponse: Executes the custom language model's API request and returns the response.
        completion(self, *args, **kwargs) -> litellm.ModelResponse: Calls complete functionality of the LLM

    Usage Example:
        # Initialize the custom LLM with the Snowflake API details
        snowflake_cortex_inference_llm = SnowflakeCortexInferenceService(model="snowflake-cortex-inference-service/snowflake-cortex-inference-service",
            model='mistral-large2',
            base_url = os.environ.get("LLM_BASE_URL"),
            api_key = os.environ.get("LLM_API_KEY"),
            snowflake_account=os.environ.get("SNOWFLAKE_ACCOUNT"),
            snowflake_service_user=os.environ.get("SNOWFLAKE_USER"),
            privatekey_password=privatekey_password, # if api_key encripted, if not None
            timeout=1800,
            stop=[], # List if words configured to stop generation
        )

        # Register the Snowflake Cortext Service custom LLM as a provider in LiteLLM
        litellm.custom_provider_map = [ # 👈 KEY STEP - REGISTER HANDLER
                {"provider": "provider-name", "custom_handler": snowflake_cortex_inference_llm}
        ]

        # Use the custom LLM to generate a response
        response = snowflake_cortex_inference_llm.completion(messages=[{"role": "user", "content": "Tell me about the ocean"}])

        # Use the liteLLM to generate a response
        response = custom_llm.call(messages=[{"role": "user", "content": "Tell me about the ocean"}])

    The SnowflakeCortexServiceLLM class is designed to integrate tightly with the Snowflake Cortex ServiceAPI,
    allowing the usage of supported LLMs models by Cortex. By adding it to the `custom_provider_map`, this custom model
    can be used within the LiteLLM framework.
    """

    def __init__(
        self,
        model: str,
        base_url: str,
        snowflake_account: Optional[str] = None,
        snowflake_service_user: Optional[str] = None,
        snowflake_authmethod: Optional[str] = "oauth",
        snowflake_token_path: Optional[str] = "/snowflake/session/token",
        privatekey_password: Optional[str] = None,
        timeout: Optional[Union[float, int]] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        n: Optional[int] = None, # TODO
        stop: Optional[Union[str, List[str]]] = None, # TODO
        max_completion_tokens: Optional[int] = None,
        api_key: Optional[str] = None,
        format_messages_callback: Optional[Any] = None,
        **kwargs,
    ):
        self.model = model
        self.base_url = base_url
        self.snowflake_account = snowflake_account
        self.snowflake_service_user = snowflake_service_user
        self.snowflake_authmethod = snowflake_authmethod
        self.snowflake_token_path = snowflake_token_path
        self.privatekey_password = privatekey_password
        self.timeout = timeout
        self.temperature = temperature
        self.top_p = top_p
        self.n = n
        self.stop = stop
        self.max_completion_tokens = max_completion_tokens
        self.api_key = api_key
        self.format_messages_callback = format_messages_callback
        self.kwargs = kwargs

        super().__init__()

    def _validate_environment(self):
        """Validates input parameters"""
        if self.snowflake_authmethod == "jwt":
            if not self.snowflake_account or not self.snowflake_service_user or not self.api_key:
                raise BaseLLMException(
                    status_code=500,
                    message="JWT auth method needs snowflake_account "
                    "snowflake_service_user and api_key to be set."
                )

    def _execute_format_callback(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Execute callback function for messages"""
        if self.format_messages_callback:
            return self.format_messages_callback(messages=messages)
        else:
            return messages

    def _get_auth_headers(self) -> Dict[str, str]:
        """
        Returns the appropriate headers based on the selected authentication method.
        For "oauth", it reads the token from '/snowflake/session/token'.
        For "jwt", it uses JWTGenerator to generate a token from the provided private key.
        """
        if self.snowflake_authmethod == "oauth" and self.snowflake_token_path:
            try:
                with open(self.snowflake_token_path, "r") as f:
                    oauth_token = f.read().strip()
            except FileNotFoundError:
                raise BaseLLMException(
                    status_code=500,
                    message=f"""OAuth token file not found at {self.snowflake_token_path}. 
                    This file is provided only inside Snowflake containers."""
                )
            headers = {
                "Authorization": f"Bearer {oauth_token}",
                "Content-Type": "application/json",
                "Accept": "application/json",
                "X-Snowflake-Authorization-Token-Type": "OAUTH",
            }
        else:  # Default to JWT (key pair) authentication
            bearerToken = JWTGenerator(
                account = self.snowflake_account, # type: ignore
                user = self.snowflake_service_user, # type: ignore
                private_key_string = self.api_key.replace("\\n", "\n").strip(), # type: ignore
                passphrase = self.privatekey_password, # type: ignore
            ).get_token()

            # Set the headers for the request
            headers = {
                "Authorization": f"Bearer {bearerToken}",
                "X-Snowflake-Authorization-Token-Type": "KEYPAIR_JWT",
                "Content-Type": "application/json",
                "Accept": "application/json, text/event-stream",
            }
        return headers
    
    def _process_response(self, response: requests.Response):
        """Process streaming response"""
        # Initialize variables to accumulate content and token counts
        accumulated_content = ""
        total_prompt_tokens = 0
        total_completion_tokens = 0
        final_response = ""
        for line in response.iter_lines():
            if line:  # Ignore empty lines
                line_str = line.decode("utf-8")

                if line_str.startswith("data:"):
                    event_data = line_str[
                        len("data:") :
                    ].strip()  # Extract event data (JSON)
                    try:
                        parsed_data = json.loads(
                            event_data
                        )  # Parse as JSON

                        # Extract content from 'choices' and accumulate it
                        delta = parsed_data.get("choices", [{}])[0].get(
                            "delta", {}
                        )
                        content = delta.get("content", "")
                        accumulated_content += content

                        # Update token usage
                        usage = parsed_data.get("usage", {})
                        total_prompt_tokens += usage.get("prompt_tokens", 0)
                        total_completion_tokens += usage.get(
                            "completion_tokens", 0
                        )

                    except json.JSONDecodeError:
                        print("Error decoding event data:", event_data)

        final_response = accumulated_content.strip()
        return total_prompt_tokens, total_completion_tokens, final_response

    def completion(self,
                   logging_obj: LiteLLMLoggingObj,
                   messages: List[Dict[str, str]]=[]) -> litellm.ModelResponse:
        """
        Handles the completion request for the custom language model via event-stream.

        This method is responsible for invoking the language model to generate completions based
        on the provided input arguments. It interfaces with the underlying model API (such as
        Snowflake API or any other custom model) to process the completion request, and returns
        the response encapsulated in a `litellm.ModelResponse`.

        Returns:
            litellm.ModelResponse: A response object containing the model's completion result,
                                  including the generated text, any choices, and other metadata
                                  associated with the completion request.
        """
        # initialize default response values
        modelResponse = None
        final_response = ""
        
        try:
            
            payload = {
                "model": self.model,
                "messages": []
            }

            if self.top_p:
                payload["top_p"] = self.top_p
            
            if self.temperature:
                payload["temperature"] = self.temperature

            messages = self._execute_format_callback(messages)
            payload["messages"] = messages

            headers = self._get_auth_headers()

            logging_obj.pre_call(
                input=messages,
                api_key=self.api_key,
                additional_args={
                    "headers": headers,
                    "api_base": self.base_url,
                    "complete_input_dict": payload,
                },
            )

            payload = json.dumps(payload)
            
            # Make the POST request to the API with streaming enabled
            response = requests.post(
                url=self.base_url,
                headers=headers,
                data=payload,
                stream=True,
                timeout=self.timeout,
            )

            total_prompt_tokens = 0
            total_completion_tokens = 0

            try:
                # Check if the response is a valid event-stream
                response_content_type = response.headers.get("Content-Type")
                if response_content_type and "text/event-stream" in response_content_type:
                    final_response = self._process_response(response)
                else:
                    # Handle non-event-stream responses
                    final_response = response.json()

                logging_obj.post_call(
                    api_key=self.api_key,
                    original_response=response,
                    additional_args={
                        "headers": headers,
                        "api_base": self.base_url,
                    },
                )

            finally:
                response.close()
        except Exception as e:
            status_code = getattr(e, "status_code", 500)
            error_headers = getattr(e, "headers", None)
            error_text = getattr(e, "text", str(e))
            error_response = getattr(e, "response", None)
            raise BaseLLMException(status_code=status_code,
                                   message=error_text,
                                   headers=error_headers,
                                   response=error_response
            )

        json_response = {
            "id": f"chatcmpl-{generate_unique_id()}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": self.model,
            "choices": [
                {
                    "finish_reason": "stop",
                    "index": 0,
                    "message": {
                        "content": str(final_response),
                        "role": "assistant",
                    },
                }
            ],
            "usage": {
                "prompt_tokens": total_prompt_tokens,
                "completion_tokens": total_completion_tokens,
                "total_tokens": total_prompt_tokens + total_completion_tokens,
            },
        }

        # Return the response data wrapped in the ModelResponse
        modelResponse = litellm.ModelResponse(
            object=json_response["object"],
            choices=json_response["choices"],
            id=json_response["id"],
            created=json_response["created"],
            model=json_response["model"],
            usage=json_response["usage"],
        )

        return modelResponse


    def streaming(self, *args, **kwargs) -> Iterator[GenericStreamingChunk]:
        raise BaseLLMException(
            status_code=500, message="Not implemented yet!"
        )

    async def acompletion(self, *args, **kwargs) -> ModelResponse:
        raise BaseLLMException(
            status_code=500, message="Not implemented yet!"
        )

    async def astreaming(self, *args, **kwargs) -> AsyncIterator[GenericStreamingChunk]:
        raise BaseLLMException(
            status_code=500, message="Not implemented yet!"
        )
