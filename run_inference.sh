#echo "### Instruction:
#Given a text, extract the gene-disease-relation triplets in a json format.
#
#### Input:
#
#In summary, the genotype distributions of the  AGT  M235T polymorphism influenced the risk of essential hypertension in south Indian women and  ACE  DD is a risk in south Indian male population. MDR analysis reveals that if both TT and DD genotypes are present, prevalence of EHT is higher in the present study. The haplotype based MDR analysis suggests that we had adequate power to detect the functional relationship of the best factor model, increasing the risk of essential hypertension associated with combined genetic variations. This is the first report to evaluate the simultaneous association of  ACE,  AGTR1, and  AGT  gene polymorphisms in essential hypertension by haplotype based analyses and gender specific association in south India. Future studies are required to consider the joint effects of several candidate genes to dissect the genetic framework and gender specific association of essential hypertension.
#
#
#### Response:" |
python recipes/quickstart/inference/local_inference/inference.py \
    --model_name 'meta-llama/Llama-3.2-3B-Instruct' \
    --peft_model './my_lora_weights/3.2-3B_biotrip_new_template' \
    --quantization '4bit' \
    --max_new_tokens 8024 \
    --top_p 1 \
    --top_k 100 \
    --temperature 0.01 \
    --share_gradio True \
    --enable_salesforce_content_safety False \
    --full_dataset \
    --dataset_mode 'val' ## for inference on bio-triplex
