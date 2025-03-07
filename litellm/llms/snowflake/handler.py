"""
SnowflakeCortexInferenceService: A custom implementation of a Snowflake Cortext
Inference Service designed to integrate with the Snowflake API and be registered
as a provider within LiteLLM.
"""

import json
import time
import requests
from typing import Any, AsyncIterator, Dict, Iterator, List, Optional, Union

import litellm
from litellm import LiteLLMLoggingObj
from litellm.llms.base import BaseLLM
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.types.utils import GenericStreamingChunk, ModelResponse
from litellm.llms.snowflake.jwt_generator import JWTGenerator


class SnowflakeCortexInferenceService(BaseLLM):
    """

    This class extends the base `CustomLLM` class to facilitate the integration with the Snowflake Cortext Inference Service API,
    allowing access to supported LLMs in Snowflake Cortex.
    See more info at https://docs.snowflake.com/en/user-guide/snowflake-cortex/llm-functions

    Attributes:
        format_messages_callback (Optional[Any]): Is a function to be called before post request in order to preprocess history messages if needed
        generate_id_callback  (Optional[Any]): Is a function to generate ids for api calls
        **kwargs,


    Methods:
        __init__(self, **kwargs): Initializes the custom LLM with the provided configuration options.
        completion(self, *args, **kwargs) -> litellm.ModelResponse: Calls complete functionality of the LLM

    Usage Example:
        # Initialize the custom LLM with the Snowflake API details
        snowflake_cortex_llm = SnowflakeCortexInferenceService()
        snowflake_cortex_llm.complete(
            model='mistral-large2',
            base_url = base_url,
            api_key = api_key,
            snowflake_account=os.environ.get("SNOWFLAKE_ACCOUNT"),
            snowflake_service_user=os.environ.get("SNOWFLAKE_USER"),
            privatekey_password=privatekey_password, # if api_key encripted, if not None
            timeout=1800,
            messages = [{"role": 'user', "content": 'What is Snowflake?'}]
        )
    """

    def __init__(self):
        self.format_messages_callback: Optional[Any] = None
        self.generate_id_callback: Optional[Any] = None
        super().__init__()

    def set_format_callback(self, format_messages_callback: Optional[Any] = None):
        self.format_messages_callback = format_messages_callback

    def set_generate_id_callback(self, generate_id_callback: Optional[Any] = None):
        self.generate_id_callback = generate_id_callback

    def _execute_format_callback(
        self, messages: List[Dict[str, str]]
    ) -> List[Dict[str, str]]:
        """Execute callback function for messages"""
        if self.format_messages_callback:
            return self.format_messages_callback(messages=messages)
        else:
            return messages

    def _get_auth_headers(
        self,
        snowflake_authmethod,
        snowflake_token_path,
        snowflake_account,
        snowflake_service_user,
        api_key,
        privatekey_password,
    ) -> Dict[str, str]:
        """
        Returns the appropriate headers based on the selected authentication method.
        For "oauth", it reads the token from '/snowflake/session/token'.
        For "jwt", it uses JWTGenerator to generate a token from the provided private key.
        """
        if snowflake_authmethod == "oauth" and snowflake_token_path:
            try:
                with open(snowflake_token_path, "r") as f:
                    oauth_token = f.read().strip()
            except FileNotFoundError:
                raise BaseLLMException(
                    status_code=500,
                    message=f"""OAuth token file not found at {snowflake_token_path}. 
                    This file is provided only inside Snowflake containers.""",
                )
            headers = {
                "Authorization": f"Bearer {oauth_token}",
                "Content-Type": "application/json",
                "Accept": "application/json",
                "X-Snowflake-Authorization-Token-Type": "OAUTH",
            }
        else:  # Default to JWT (key pair) authentication
            bearerToken = JWTGenerator(
                account=snowflake_account,  # type: ignore
                user=snowflake_service_user,  # type: ignore
                private_key_string=api_key.replace("\\n", "\n").strip(),  # type: ignore
                passphrase=privatekey_password,  # type: ignore
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
                        parsed_data = json.loads(event_data)  # Parse as JSON

                        # Extract content from 'choices' and accumulate it
                        delta = parsed_data.get("choices", [{}])[0].get("delta", {})
                        content = delta.get("content", "")
                        accumulated_content += content

                        # Update token usage
                        usage = parsed_data.get("usage", {})
                        total_prompt_tokens += usage.get("prompt_tokens", 0)
                        total_completion_tokens += usage.get("completion_tokens", 0)

                    except json.JSONDecodeError:
                        print("Error decoding event data:", event_data)

        final_response = accumulated_content.strip()
        return total_prompt_tokens, total_completion_tokens, final_response

    def completion(
        self,
        model: str,
        base_url: str,
        logging_obj: LiteLLMLoggingObj,
        snowflake_account: Optional[str] = None,
        snowflake_service_user: Optional[str] = None,
        snowflake_authmethod: Optional[str] = "oauth",
        snowflake_token_path: Optional[str] = "/snowflake/session/token",
        privatekey_password: Optional[str] = None,
        timeout: Optional[Union[float, int]] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        api_key: Optional[str] = None,
        messages: List[Dict[str, str]] = [],
    ) -> litellm.ModelResponse:
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

        """Validates input parameters"""
        if snowflake_authmethod == "jwt":
            if not snowflake_account or not snowflake_service_user or not api_key:
                raise BaseLLMException(
                    status_code=500,
                    message="JWT auth method needs snowflake_account "
                    "snowflake_service_user and api_key to be set.",
                )

        api_key = api_key.replace("\\n", "\n").strip()  # type: ignore
        # initialize default response values
        modelResponse = None
        final_response = ""

        try:

            payload = {"model": model, "messages": []}

            if top_p:
                payload["top_p"] = top_p

            if temperature:
                payload["temperature"] = temperature

            messages = self._execute_format_callback(messages)
            payload["messages"] = messages

            headers = self._get_auth_headers(
                snowflake_authmethod,
                snowflake_token_path,
                snowflake_account,
                snowflake_service_user,
                api_key,
                privatekey_password,
            )
            if logging_obj:
                logging_obj.pre_call(
                    input=messages,
                    api_key=api_key,
                    additional_args={
                        "headers": headers,
                        "api_base": base_url,
                        "complete_input_dict": payload,
                    },
                )

            payload = json.dumps(payload)

            # Make the POST request to the API with streaming enabled
            response = requests.post(
                url=base_url,
                headers=headers,
                data=payload,
                stream=True,
                timeout=timeout,
            )

            total_prompt_tokens = 0
            total_completion_tokens = 0

            try:
                # Check if the response is a valid event-stream
                response_content_type = response.headers.get("Content-Type")
                if (
                    response_content_type
                    and "text/event-stream" in response_content_type
                ):
                    final_response = self._process_response(response)
                else:
                    # Handle non-event-stream responses
                    final_response = response.json()

                if logging_obj:
                    logging_obj.post_call(
                        api_key=api_key,
                        original_response=response,
                        additional_args={
                            "headers": headers,
                            "api_base": base_url,
                        },
                    )

            finally:
                response.close()
        except Exception as e:
            status_code = getattr(e, "status_code", 500)
            error_headers = getattr(e, "headers", None)
            error_text = getattr(e, "text", str(e))
            error_response = getattr(e, "response", None)
            raise BaseLLMException(
                status_code=status_code,
                message=error_text,
                headers=error_headers,
                response=error_response,
            )
        identifier = ""
        if self.generate_id_callback:
            identifier = self.generate_id_callback()
        else:
            identifier = str(int(time.time() * 1000))
        json_response = {
            "id": f"chatcmpl-{identifier}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
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
        raise BaseLLMException(status_code=500, message="Not implemented yet!")

    async def acompletion(self, *args, **kwargs) -> ModelResponse:
        raise BaseLLMException(status_code=500, message="Not implemented yet!")

    async def astreaming(self, *args, **kwargs) -> AsyncIterator[GenericStreamingChunk]:
        raise BaseLLMException(status_code=500, message="Not implemented yet!")
