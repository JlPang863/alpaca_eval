# export CUDA_VISIBLE_DEVICES=7
# NUM_GPUS=1

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
NUM_GPUS=8

main_process_port=29517
root_path="/mnt/data1/jinlong/CL_DPO_outputs"

BATCH_SIZE=4
OUTPUT_PATH="./model_outputs_cl"

BASE_MODELS=(
    # llama-3-8b
    mistral-7b
)

# model_type='instruct'
model_type='base'
loss_type_list=(
    sft-chosen
    # org
    # dpo
    # simpo
    # selective-dpo
    # mix-dpo
    # dpo-identical-pairs-high-quality-2000
    # dpo-identical-pairs-low-quality-2000
    ) 



for base_model in ${BASE_MODELS[@]}; do

    for loss_type in ${loss_type_list[@]}; do

        if [[ $loss_type == "org" ]]; then
            if [[ $base_model == "llama-3-8b" ]]; then
                model_name_or_path="meta-llama/Meta-Llama-3-8B-Instruct"
            elif [[ $base_model == "mistral-7b" ]]; then
                model_name_or_path="mistralai/Mistral-7B-Instruct-v0.2"
            else
                echo "unknown base model"
            fi
        else
            model_name_or_path="${root_path}/${base_model}-${model_type}-${loss_type}"
        fi
        echo "*** model_name_or_path: ${model_name_or_path} ***"

        accelerate launch \
            --num_processes $NUM_GPUS \
            --main_process_port $main_process_port \
            --mixed_precision bf16 \
            generate_response.py \
            --model_name_or_path $model_name_or_path \
            --batch_size $BATCH_SIZE \
            --output_path $OUTPUT_PATH \

    done

done 

