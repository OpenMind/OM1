# Deprecated: Use NearAI or Open AI LLM plugins instead.

# import logging
# import time
# import typing as T

# import requests
# from pydantic import BaseModel

# from llm import LLM, LLMConfig

# R = T.TypeVar("R", bound=BaseModel)


# class MultiLLM(LLM[R]):
#     """
#     MultiLLM implementation that sends requests to the robotic team endpoint.

#     This plugin maintains the same output structure as other LLM plugins
#     while routing requests through the agent-based robotic team.

#     Parameters
#     ----------
#     output_model : Type[R]
#         A Pydantic BaseModel subclass defining the expected response structure.
#     config : LLMConfig
#         Configuration object containing API settings. If not provided, defaults
#         will be used.
#     """

#     def __init__(self, output_model: T.Type[R], config: LLMConfig = LLMConfig()):
#         """
#         Initialize the MultiLLM instance.

#         Parameters
#         ----------
#         output_model : Type[R]
#             Pydantic model class for response validation.
#         config : LLMConfig, optional
#             Configuration settings for the LLM.
#         """
#         super().__init__(output_model, config)

#         if not config.api_key:
#             raise ValueError("config file missing api_key")

#         if not config.model:
#             self._config.model = "gpt-4.1-nano"

#         self.endpoint = "https://api.openmind.org/api/core/agent"

#     async def ask(
#         self, prompt: str, messages: T.List[T.Dict[str, str]] = []
#     ) -> R | None:
#         """
#         Send a prompt to the appropriate endpoint and get a structured response.
#         If RAG is enabled, it will query the knowledge base first.

#         Parameters
#         ----------
#         prompt : str
#             The input prompt to send to the model.
#         messages : List[Dict[str, str]]
#             Message history to provide context.

#         Returns
#         -------
#         R or None
#             Parsed response matching the output_model structure, or None if
#             parsing fails.
#         """
#         try:
#             self.io_provider.llm_start_time = time.time()
#             self.io_provider.set_llm_prompt(prompt)

#             headers = {
#                 "Authorization": f"Bearer {self._config.api_key}",
#                 "Content-Type": "application/json",
#             }

#             request = {
#                 "system_prompt": self.io_provider.fuser_system_prompt,
#                 "inputs": self.io_provider.fuser_inputs,
#                 "available_actions": self.io_provider.fuser_available_actions,
#                 "model": self._config.model,
#                 "response_format": self._output_model.model_json_schema(),
#                 "structured_outputs": True,
#             }

#             logging.debug(f"MultiLLM system_prompt: {request['system_prompt']}")
#             logging.debug(f"MultiLLM inputs: {request['inputs']}")
#             logging.debug(f"MultiLLM available_actions: {request['available_actions']}")

#             response = requests.post(
#                 self.endpoint,
#                 json=request,
#                 headers=headers,
#             )

#             response_json = response.json()
#             self.io_provider.llm_end_time = time.time()
#             logging.info(f"Raw response: {response_json}")

#             output = response_json.get("content")
#             try:
#                 parsed_response = self._output_model.model_validate_json(output)
#                 logging.debug(f"MultiLLM structured output: {parsed_response}")
#                 return parsed_response
#             except Exception as e:
#                 logging.error(f"Error validating structured response: {e}")
#                 return None

#         except Exception as e:
#             logging.error(f"Error during API request: {str(e)}")
#             return None

#     async def close(self):
#         """Close the session when done."""
#         if self.session:
#             await self.session.close()
#             self.session = None
