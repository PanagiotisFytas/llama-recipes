python recipes/quickstart/inference/local_inference/inference.py \
    --model_name 'meta-llama/Llama-3.2-1B-Instruct' \
    --peft_model './my_lora_weights/biotrip_json_no_quant_weight_decay' \
    --max_new_tokens 512 \
    --prompt_file './test_biotriplex_prompt.txt'