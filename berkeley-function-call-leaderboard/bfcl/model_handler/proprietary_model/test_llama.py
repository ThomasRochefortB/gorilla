import os
import json
from openai import OpenAI
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from bfcl.model_handler.base_handler import BaseHandler
from bfcl.model_handler.model_style import ModelStyle
from bfcl.model_handler.utils import (
    func_doc_language_specific_pre_processing,
    system_prompt_pre_processing_chat_model,
    default_decode_ast_prompting,
    default_decode_execute_prompting,

)

class LocalLlamaHandler(BaseHandler):
    def __init__(self, model_name, temperature) -> None:
        super().__init__(model_name, temperature)
        self.model_style = ModelStyle.OSSMODEL
        
        # Read from env vars with fallbacks
        vllm_host = os.getenv('VLLM_ENDPOINT', 'localhost')
        vllm_port = os.getenv('VLLM_PORT', '8000')
        
        # Construct the API base URL
        api_base_url = f"http://{vllm_host}:{vllm_port}/v1"
        
        self.client = OpenAI(base_url=api_base_url, api_key="EMPTY")
        self.model_name_huggingface = model_name.replace("-FC", "")
        self.temperature = 0.0
    def _format_prompt(self, messages, function):
        formatted_prompt = "<|begin_of_text|>"

        for message in messages:
            formatted_prompt += f"<|start_header_id|>{message['role']}<|end_header_id|>\n\n{message['content'].strip()}<|eot_id|>"

        formatted_prompt += f"<|start_header_id|>assistant<|end_header_id|>\n\n"

        return formatted_prompt
    

    def decode_ast(self, result, language="Python"):
        return default_decode_ast_prompting(result, language)

    def decode_execute(self, result):
        return default_decode_execute_prompting(result)

    

    def _query_prompting(self, inference_data: dict):
        # We use the OpenAI Completions API
        function: list[dict] = inference_data["function"]
        message: list[dict] = inference_data["message"]

        formatted_prompt: str = self._format_prompt(message, function)
        inference_data["inference_input_log"] = {"formatted_prompt": formatted_prompt}

        if hasattr(self, "stop_token_ids"):
            api_response = self.client.completions.create(
                model=self.model_name_huggingface,
                temperature=self.temperature,
                prompt=formatted_prompt,
                stop_token_ids=self.stop_token_ids,
                max_tokens=4096,  # TODO: Is there a better way to handle this?
            )
        else:
            api_response = self.client.completions.create(
                model=self.model_name_huggingface,
                temperature=self.temperature,
                prompt=formatted_prompt,
                max_tokens=4096,
            )

        return api_response

    def _pre_query_processing_prompting(self, test_entry: dict) -> dict:
        functions: list = test_entry["function"]
        test_category: str = test_entry["id"].rsplit("_", 1)[0]

        functions = func_doc_language_specific_pre_processing(functions, test_category)

        test_entry["question"][0] = system_prompt_pre_processing_chat_model(
            test_entry["question"][0], functions, test_category
        )

        return {"message": [], "function": functions}

    def _parse_query_response_prompting(self, api_response: any) -> dict:
        return {
            "model_responses": api_response.choices[0].text,
            "input_token": api_response.usage.prompt_tokens,
            "output_token": api_response.usage.completion_tokens,
        }

    def add_first_turn_message_prompting(
        self, inference_data: dict, first_turn_message: list[dict]
    ) -> dict:
        inference_data["message"].extend(first_turn_message)
        return inference_data

    def _add_next_turn_user_message_prompting(
        self, inference_data: dict, user_message: list[dict]
    ) -> dict:
        inference_data["message"].extend(user_message)
        return inference_data

    def _add_assistant_message_prompting(
        self, inference_data: dict, model_response_data: dict
    ) -> dict:
        inference_data["message"].append(
            {"role": "assistant", "content": model_response_data["model_responses"]}
        )
        return inference_data

    def _add_execution_results_prompting(
        self, inference_data: dict, execution_results: list[str], model_response_data: dict
    ) -> dict:
        for execution_result, decoded_model_response in zip(
            execution_results, model_response_data["model_responses_decoded"]
        ):
            inference_data["message"].append(
                {
                    "role": "tool",
                    "name": decoded_model_response,
                    "content": execution_result,
                }
            )
        return inference_data

    def inference(self, test_entry: dict, include_input_log: bool, include_state_log: bool):
        try:
            if "multi_turn" in test_entry["id"]:
                model_responses, metadata = self.inference_multi_turn_prompting(test_entry, include_input_log, include_state_log)
            else:
                model_responses, metadata = self.inference_single_turn_prompting(test_entry, test_entry, include_input_log)
        except Exception as e:
            print("-" * 100)
            print(
                "❗️❗️ Error occurred during inference. Maximum reties reached for rate limit or other error. Continuing to next test case."
            )
            print(f"❗️❗️ Test case ID: {test_entry['id']}, Error: {str(e)}")
            print("-" * 100)

            model_responses = f"Error during inference: {str(e)}"
            metadata = {}

        return model_responses, metadata

    def batch_inference(
        self,
        test_entries: list[dict],
        num_gpus: int,
        gpu_memory_utilization: float,
        backend: str,
        include_input_log: bool,
        include_state_log: bool,
    ):
        """
        Batch inference for OSS models.
        """
        

        

        try:
            
            futures = []
            with ThreadPoolExecutor(max_workers=100) as executor:
                with tqdm(
                    total=len(test_entries),
                    desc=f"Generating results for {self.model_name}",
                ) as pbar:

                    for test_case in test_entries:
                        future = executor.submit(self._multi_threaded_inference, test_case, include_input_log, include_state_log)
                        futures.append(future)

                    for future in futures:
                        # This will wait for the task to complete, so that we are always writing in order
                        result = future.result()
                        self.write(result)
                        pbar.update()


        except Exception as e:
            raise e


            
    def _multi_threaded_inference(self, test_case, include_input_log: bool, include_state_log: bool):
        """
        This is a wrapper function to make sure that, if an error occurs during inference, the process does not stop.
        """
        assert type(test_case["function"]) is list

        try:
            if "multi_turn" in test_case["id"]:
                model_responses, metadata = self.inference_multi_turn_prompting(test_case,include_input_log, include_state_log)
            else:
                model_responses, metadata = self.inference_single_turn_prompting(test_case, include_input_log)
        except Exception as e:
            print("-" * 100)
            print(
                "❗️❗️ Error occurred during inference. Maximum reties reached for rate limit or other error. Continuing to next test case."
            )
            print(f"❗️❗️ Test case ID: {test_case['id']}, Error: {str(e)}")
            print("-" * 100)

            model_responses = f"Error during inference: {str(e)}"

        return {
            "id": test_case["id"],
            "result": model_responses,
        }
