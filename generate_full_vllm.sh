# export CUDA_VISIBLE_DEVICES=7
# NUM_GPUS=1

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
NUM_GPUS=8

main_process_port=29517
root_path="/mnt/data1/jinlong/DPO-noisy-outputs/"

BATCH_SIZE=16
OUTPUT_PATH="./model_outputs"

# base_model="mistral-7b" #llama-3-8b mistral-7b
base_model="llama-3-8b"
# loss_type_list=("sft" "dpo" "cdpo" "robust" "ipo" "dpj")

# loss_type_list=("dpo-new" "cdpo-new" "robust-new" "ours1-1") #"dpo-new" "cdpo-new" "robust-new" 

# loss_type_list=("sft")
# loss_type_list=("dpo-new1" "cdpo-new1" "robust-new1" "ours1-1-new1")
# loss_type_list=("robust-new1")
# loss_type_list=("kto" "ours4-1-new1") 

# loss_type_list=("kto-filtered") 
# loss_type_list=("ours1-3-new1" "ours4-2-new1" "dpo-mean-new1") 
# loss_type_list=("dpo-random" "dpo-top" "dpo-bottom") 
# loss_type_list=("dpo-new1-sorted-llama-full" "dpo-new1-sorted-score-diff-full" "dpo-new1-sorted-llama-half"  "dpo-new1-sorted-score-diff-half") #"dpo-new1-sorted-llama-full" "dpo-new1-sorted-score-diff-full" "dpo-new1-sorted-llama-half"  "dpo-new1-sorted-score-diff-half"
loss_type_list=("dpo-random-identical-reverse" "dpo-random-identical") #
loss_type_list=("dpo-random-identical-reverse") #


for loss_type in ${loss_type_list[@]}; do

    # model_name_or_path="/mnt/data1/jinlong/DPO-noisy-outputs/llama-3-1b-sft"

    ### model path ###
    if [[ $loss_type == "sft" ]]; then
        if [[ $base_model == "llama-3-8b" ]]; then
            model_name_or_path="princeton-nlp/Llama-3-Base-8B-SFT"
        elif [[ $base_model == "mistral-7b" ]]; then
            model_name_or_path="alignment-handbook/zephyr-7b-sft-full"
        else
            model_name_or_path="${root_path}/${base_model}-sft"
        fi
    else
        model_name_or_path="${root_path}/${base_model}-${loss_type}"
    fi

    accelerate launch \
        --num_processes $NUM_GPUS \
        --main_process_port $main_process_port \
        --mixed_precision bf16 \
        generate_response_vllm.py \
        --model_name_or_path $model_name_or_path \
        --batch_size $BATCH_SIZE \
        --output_path $OUTPUT_PATH \
        --use_vllm=True

done

