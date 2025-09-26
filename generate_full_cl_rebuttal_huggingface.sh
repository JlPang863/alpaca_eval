# export CUDA_VISIBLE_DEVICES=7
# NUM_GPUS=1

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
NUM_GPUS=8

main_process_port=29517
root_path="/home/azureuser/cloudfiles/code/Users/jinlong.pang/QualityDPO/CL_DPO_outputs"

BATCH_SIZE=8
OUTPUT_PATH="./model_outputs_cl_instruct"


MODEL_LIST=(
    # llama-3-8b-instruct-mix-dpo
    # llama-3-8b-instruct-mix-dpo-loss
    # llama-3-8b-instruct-mix-dpo-rm
    # llama-3-8b-instruct-selective-dpo
    # princeton-nlp/Llama-3-Instruct-8B-DPO
    # mistral-7b-instruct-mix-dpo
    # mistral-7b-instruct-mix-dpo-loss
    # mistral-7b-instruct-mix-dpo-rm
    # mistral-7b-instruct-selective-dpo
    # mistral-7b-instruct-mix-dpo-lr1
    # mistral-7b-instruct-mix-dpo-lr2
    # llama-3-8b-base-mix-dpo-5-swap
    # mistral-7b-instruct-mix-dpo-lr3
    # mistral-7b-instruct-mix-dpo-lr4
    # llama-3-8b-base-cal-dpo
    # mistral-7b-base-cal-dpo
    # llama-3-8b-base-dpo-ches-0.5
    # llama-3-8b-base-dpo-ches-0.9
    # llama-3-8b-base-dpo-nll
    mistral-7b-base-dpo-nll
    # mistral-7b-base-dpo-ches-0.5
    # mistral-7b-base-dpo-ches-0.9

)


for model_name_or_path in ${MODEL_LIST[@]}; do

        echo "*** model_name_or_path: ${model_name_or_path} ***"
        accelerate launch \
            --num_processes $NUM_GPUS \
            --main_process_port $main_process_port \
            --mixed_precision bf16 \
            generate_response.py \
            --model_name_or_path $root_path/$model_name_or_path \
            --batch_size $BATCH_SIZE \
            --output_path $OUTPUT_PATH \

done 

# huggingface-cli download princeton-nlp/Llama-3-Instruct-8B-DPO --local-dir /home/azureuser/cloudfiles/code/Users/jinlong.pang/QualityDPO/CL_DPO_outputs