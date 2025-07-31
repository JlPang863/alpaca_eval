# export CUDA_VISIBLE_DEVICES=7
# NUM_GPUS=1

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
NUM_GPUS=8

main_process_port=29515
root_path="/mnt/data1/jinlong/CL_DPO_outputs"

BATCH_SIZE=16
OUTPUT_PATH="./model_outputs_cl"

# base_model="mistral-7b" #llama-3-8b mistral-7b
# base_model="llama-3-8b"

base_model="qwen-2.5-7b"
loss_type_list=(
    # sft
    # selectiveDPO
    # dpo-full
    # dpo-sorted-qwen-full
    # ours4-6-sorted-score-diff-full
    # simpo-full
    # ours4-6-sorted-score-diff-full-lr1
    # simpo-full-new1
    # ours4-6-sorted-score-diff-full-lr3-ckpt-382
    ours4-6-sorted-score-diff-full-lr2
    ours4-6-sorted-score-diff-full-lr3
)


for loss_type in ${loss_type_list[@]}; do
    # model_name_or_path="/mnt/data1/jinlong/DPO-noisy-outputs/llama-3-1b-sft"

    echo "*** loss type: $loss_type ***"
    if [[ $base_model == "llama-3-8b" ]]; then
        base_model_name='Llama-3-Base-8B-SFT'
    elif [[ $base_model == "mistral-7b" ]]; then
        base_model_name='Mistral-7B-Base-SFT'
    fi

    ## model path ###
    if [[ $loss_type == "sft" ]]; then
        if [[ $base_model == "llama-3-8b" ]]; then
            model_name_or_path="princeton-nlp/Llama-3-Base-8B-SFT"
        elif [[ $base_model == "mistral-7b" ]]; then
            # model_name_or_path="alignment-handbook/zephyr-7b-sft-full"
            model_name_or_path="HuggingFaceH4/mistral-7b-sft-beta"
        elif [[ $base_model == "qwen-2.5-7b" ]]; then
            model_name_or_path="AmberYifan/Qwen2.5-7B-sft-ultrachat"
        else
            model_name_or_path="${root_path}/${base_model}-sft"
        fi
    elif [[ $loss_type == "dpo" ]]; then
        model_name_or_path="princeton-nlp/${base_model_name}-DPO"
    elif [[ $loss_type == "simpo" ]]; then
        if [[ $base_model == "llama-3-8b" ]]; then
            model_name_or_path="jlpang888/${base_model_name}-SimPO"
        elif [[ $base_model == "mistral-7b" ]]; then
            model_name_or_path="princeton-nlp/${base_model_name}-SimPO"
        fi
    elif [[ $loss_type == "ipo" ]]; then
        model_name_or_path="princeton-nlp/${base_model_name}-IPO"
    elif [[ $loss_type == "kto" ]]; then
        model_name_or_path="princeton-nlp/${base_model_name}-KTO"
    elif [[ $loss_type == "cpo" ]]; then
        model_name_or_path="princeton-nlp/${base_model_name}-CPO"
    elif [[ $loss_type == "rdpo" ]]; then
        model_name_or_path="princeton-nlp/${base_model_name}-RDPO"
    elif [[ $loss_type == "selectiveDPO" ]]; then
        if [[ $base_model == "llama-3-8b" ]]; then
            model_name_or_path="glorgao/SelectiveDPO-Llama3-8B-SFT-UFBinarized"
        elif [[ $base_model == "mistral-7b" ]]; then
            model_name_or_path="glorgao/SelectiveDPO-Mistral-7B-SFT-UFBinarized"
        elif [[ $base_model == "qwen-2.5-7b" ]]; then
            model_name_or_path="glorgao/SelectiveDPO-Qwen2.5-7B-SFT-UFBinarized"
        fi
    elif [[ $loss_type == "orpo" ]]; then
        if [[ $base_model == "mistral-7b" ]]; then
            model_name_or_path="kaist-ai/mistral-orpo-beta"
        else
            model_name_or_path="princeton-nlp/${base_model_name}-ORPO"
        fi
    elif [[ $loss_type == "slic-hf" ]]; then
        model_name_or_path="princeton-nlp/${base_model_name}-SLiC-HF"
    else
        model_name_or_path="${root_path}/${base_model}-${loss_type}"
    fi
    # model_name_or_path="${root_path}/${loss_type}"

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

