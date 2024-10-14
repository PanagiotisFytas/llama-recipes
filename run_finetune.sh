python recipes/quickstart/finetuning/finetuning.py \
    --use_peft \
    --peft_method lora \
    --model_name 'meta-llama/Llama-3.2-1B-Instruct' \
    --output_dir './my_lora_weights/biotrip_json_no_quant_weight_decay' \
    --batch_size_training 1 \
    --weight_decay 0.1 \
    --num_epochs 6 \
    --dataset biotriplex_dataset