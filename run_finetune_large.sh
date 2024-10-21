python recipes/quickstart/finetuning/finetuning.py \
    --use_peft \
    --peft_method lora \
    --model_name 'meta-llama/Llama-3.1-8B-Instruct' \
    --output_dir './my_lora_weights/3.1_large_biotrip_json_no_quant_high_weight_decay_long_context' \
    --batch_size_training 1 \
    --batching_strategy "padding" \
    --weight_decay 0.2 \
    --num_epochs 6 \
    --dataset biotriplex_dataset \
    --context_length 21000 \
    --quantization '4bit'



    # best val loss is 0.0556