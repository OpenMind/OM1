# import logging
# import time
# import typing as T

# import requests
# from pydantic import BaseModel

# from llm import LLM, LLMConfig

# R = T.TypeVar("R", bound=BaseModel)


# class RagMultiLLM(LLM[R]):
#     """
#     RagMultiLLM implementation that sends requests to the robotic team endpoint.

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
#         self.rag_endpoint = "https://api.openmind.org/api/core/rag/query"

#         self.use_rag = hasattr(self._config, "use_rag") and self._config.use_rag

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

#             recent_voice = ""
#             rag_context = ""
#             tools_summary = ""
#             if self.io_provider.inputs.get("Voice", None):
#                 recent_voice = self.io_provider.inputs["Voice"].input

#             if self.use_rag and recent_voice:
#                 try:
#                     rag_request = {"query": recent_voice, "skip_cache": False}

#                     logging.debug(f"Sending RAG request to {self.rag_endpoint}")
#                     rag_response = requests.post(
#                         self.rag_endpoint,
#                         json=rag_request,
#                         headers=headers,
#                     )

#                     logging.debug(f"RAG response status: {rag_response.status_code}")
#                     if rag_response.status_code == 200:
#                         rag_data = rag_response.json()
#                         logging.debug(f"RAG response data: {rag_data}")
#                         if rag_data.get("success") and "data" in rag_data:
#                             rag_content = rag_data["data"].get("content", "")
#                             rag_tools = rag_data["data"].get("tools", [])

#                             if rag_tools:
#                                 logging.info(f"RAG tools data: {rag_tools}")
#                                 tools_lines = [
#                                     "\n\nThe following tools were used to gather this information:"
#                                 ]
#                                 tools_lines += [
#                                     f"- {tool.get('tool_name', 'Unknown tool')}"
#                                     for tool in rag_tools
#                                 ]
#                                 tools_summary = "\n".join(tools_lines)

#                             if rag_content:
#                                 rag_context = rag_content.strip()
#                                 logging.info(f"RAG context added: {rag_context}")

#                 except Exception as e:
#                     logging.error(f"Error querying RAG endpoint: {str(e)}")

#             request = {
#                 "system_prompt": self.io_provider.fuser_system_prompt,
#                 "inputs": self.io_provider.fuser_inputs,
#                 "available_actions": self.io_provider.fuser_available_actions,
#                 "model": self._config.model,
#                 "response_format": self._output_model.model_json_schema(),
#                 "structured_outputs": True,
#             }

#             if rag_context:
#                 kb_block = (
#                     "\n\n--- KNOWLEDGE BASE CONTEXT ---\n" f"{rag_context}\n" "---\n"
#                 )

#                 if tools_summary:
#                     kb_block += f"{tools_summary}\n"

#                 existing_inputs = request.get("inputs") or ""
#                 request["inputs"] = f"{existing_inputs}{kb_block}".strip()

#             if rag_context:
#                 request["system_prompt"] = (
#                     f"{request['system_prompt']}\n\n"
#                     "You are provided with a 'KNOWLEDGE BASE CONTEXT' section inside "
#                     "the user inputs. Consult it when crafting your answer."
#                 )

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
