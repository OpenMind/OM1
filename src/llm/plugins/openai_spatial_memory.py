# import json
# import logging
# import time
# import typing as T

# import openai
# from pydantic import BaseModel

# from llm import LLM, LLMConfig
# from providers.unitree_go2_location_provider import UnitreeGo2LocationProvider

# R = T.TypeVar("R", bound=BaseModel)


# class OpenAISpatialMemoryLLM(LLM[R]):
#     """
#     An OpenAI-based Language Learning Model implementation.
#     This class implements the LLM interface for OpenAI's GPT models, handling
#     configuration, authentication, and async API communication.
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
#         Initialize the OpenAI LLM instance.
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
#             self._config.model = "gpt-4o-mini"

#         self._client = openai.AsyncClient(
#             base_url=config.base_url or "https://api.openmind.org/api/core/openai",
#             api_key=config.api_key,
#         )

#         self.unitree_go2_location_provider = UnitreeGo2LocationProvider()
#         self.available_functions = (
#             self.unitree_go2_location_provider.generate_llm_functions()
#         )
#         self.function_mapping = (
#             self.unitree_go2_location_provider.get_llm_function_mapping()
#         )

#     async def _execute_function(self, function_name: str, arguments: T.Dict) -> T.Dict:
#         """
#         Execute a function call using the generated mapping.
#         Parameters
#         ----------
#         function_name : str
#             The name of the function to execute.
#         arguments : Dict
#             Arguments to pass to the function.
#         Returns
#         -------
#         Dict
#             Result of the function execution, including success status and message.
#         """
#         try:
#             if function_name in self.function_mapping:
#                 method = self.function_mapping[function_name]
#                 return method(**arguments)
#             else:
#                 return {
#                     "success": False,
#                     "message": f"Unknown function: {function_name}",
#                 }
#         except Exception as e:
#             logging.error(f"Error executing function {function_name}: {e}")
#             return {"success": False, "message": f"Error executing function: {str(e)}"}

#     async def ask(
#         self, prompt: str, messages: T.List[T.Dict[str, str]] = []
#     ) -> R | None:
#         """
#         Send a prompt to the OpenAI API and get a structured response.
#         Parameters
#         ----------
#         prompt : str
#             The input prompt to send to the model.
#         messages : List[Dict[str, str]]
#             List of message dictionaries to send to the model.
#         Returns
#         -------
#         R or None
#             Parsed response matching the output_model structure, or None if
#             parsing fails.
#         """
#         try:
#             # Add function calls to the prompt
#             if self.available_functions:
#                 prompt += "\n\nAVAILABLE FUNCTIONS:\n"
#                 for func in self.available_functions.values():
#                     prompt += f'- {func["function"]["name"]}: {func["function"]["description"]}\n'
#                 available_locations = (
#                     self.unitree_go2_location_provider.list_location_names()
#                 )
#                 if available_locations:
#                     prompt += "\nAVAILABLE LOCATIONS:\n"
#                     for loc in available_locations["location_names"]:
#                         prompt += f"- {loc}\n"

#                 prompt += (
#                     "\nPlease use the available functions to follow the instructions."
#                 )

#             logging.info(f"OpenAI LLM input: {prompt}")
#             logging.info(f"OpenAI LLM messages: {messages}")

#             self.io_provider.llm_start_time = time.time()

#             # this saves all the input information
#             self.io_provider.set_llm_prompt(prompt)

#             response = await self._client.beta.chat.completions.parse(
#                 model=self._config.model,
#                 messages=[*messages, {"role": "user", "content": prompt}],
#                 timeout=self._config.timeout,
#                 tools=list(self.available_functions.values()),
#                 tool_choice="required",
#                 response_format=self._output_model,
#             )
#             logging.info(f"OpenAI LLM response: {response}")

#             for tool_call in response.choices[0].message.tool_calls:
#                 function_name = tool_call.function.name
#                 function_args = json.loads(tool_call.function.arguments)

#                 logging.info(
#                     f"Function call detected: {function_name} with args {function_args}"
#                 )
#                 function_result = await self._execute_function(
#                     function_name, function_args
#                 )
#                 logging.info(f"Function result: {function_result}")

#                 prompt += f"\n\nFunction {function_name} returned: {function_result}"

#                 prompt += (
#                     "\n\nBased on the function result, please provide a final response."
#                 )

#             response = await self._client.beta.chat.completions.parse(
#                 model=self._config.model,
#                 messages=[*messages, {"role": "user", "content": prompt}],
#                 response_format=self._output_model,
#                 timeout=self._config.timeout,
#             )

#             message_content = response.choices[0].message.content
#             self.io_provider.llm_end_time = time.time()

#             try:
#                 parsed_response = self._output_model.model_validate_json(
#                     message_content
#                 )
#                 logging.info(f"OpenAI LLM output: {parsed_response}")
#                 return parsed_response
#             except Exception as e:
#                 logging.error(f"Error parsing OpenAI response: {e}")
#                 return None

#         except Exception as e:
#             logging.error(f"OpenAI API error: {e}")
#             return None
