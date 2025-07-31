# export CUDA_VISIBLE_DEVICES=7
# NUM_GPUS=1

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
NUM_GPUS=8

main_process_port=29517
root_path="/mnt/data1/jinlong/CL_DPO_outputs"

BATCH_SIZE=4
OUTPUT_PATH="./model_outputs_cl"


MODEL_LIST=(
    princeton-nlp/Llama-3-Instruct-8B-DPO
    # princeton-nlp/Llama-3-Instruct-8B-SimPO
    # princeton-nlp/Mistral-7B-Instruct-DPO
    # princeton-nlp/Mistral-7B-Instruct-SimPO
)


for model_name_or_path in ${MODEL_LIST[@]}; do

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

