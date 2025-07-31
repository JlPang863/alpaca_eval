# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# NUM_GPUS=8

export CUDA_VISIBLE_DEVICES=6,7
NUM_GPUS=2

main_process_port=29517
root_path="/mnt/data1/jinlong/DPO-noisy-outputs/"

BATCH_SIZE=32
OUTPUT_PATH="./model_outputs"
base_model="llama-3-8b" #llama-3-8b mistral-7b

# loss_type_list=("sft" "dpo" "cdpo" "robust" "ipo" "dpj")

# loss_type_list=("dpo-new" "cdpo-new" "robust-new" "ours1-1") #"dpo-new" "cdpo-new" "robust-new" 

# loss_type_list=(""ours5-new1"")
loss_type_list=("ours1-2-new1") 

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
        model_name_or_path="${root_path}/${base_model}-${loss_type}-merged"
    fi

    accelerate launch \
        --num_processes $NUM_GPUS \
        --main_process_port $main_process_port \
        --mixed_precision bf16 \
        generate_response.py \
        --model_name_or_path $model_name_or_path \
        --batch_size $BATCH_SIZE \
        --output_path $OUTPUT_PATH \

done

