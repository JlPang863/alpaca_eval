# export CUDA_VISIBLE_DEVICES=7
# NUM_GPUS=1

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
NUM_GPUS=8

main_process_port=29517
root_path="/mnt/data1/jinlong/CL_DPO_outputs"

BATCH_SIZE=16
OUTPUT_PATH="./model_outputs_cl"

# base_model="mistral-7b" #llama-3-8b mistral-7b
base_model="llama-3-8b"

## baselises
# loss_type_list=(orpo rdpo slic-hf dpo cpo ipo simpo kto sft)
loss_type_list=(simpo)
# loss_type_list=(
    # ours4-6-sorted-score-diff-full-filter-out-similar-samples-ckpt-336
    # ours4-6-sorted-score-diff-full-filter-out-similar-samples
    # ours4-6-sorted-score-diff-full-shuffle
    # ours4-6-sorted-score-diff-full-lr1
    # ours4-6-sorted-llama-full
    # ours4-6-sorted-score-diff-full
    # ours4-6-sorted-score-diff-full-lr1-ckpt-336
    

    # )
# loss_type_list=(
#     # dpo-sorted-llama-full
#     # dpo-sorted-reward-diff-full
#     # dpo-sorted-score-diff-full
#     # dpo-sorted-llama-full-ckpt-191
#     # dpo-sorted-reward-diff-full-ckpt-191
#     # dpo-sorted-score-diff-full-ckpt-191

#     # dpo-sorted-docta-score-diff-full
#     # dpo-sorted-embedding-distance-full
#     # dpo-sorted-embedding-distance-full-ckpt-191
#     # ours4-6-sorted-embedding-distance-full
#     # ours4-6-sorted-score-diff-full
#     # ours4-8-sorted-reward-diff-full
#     # ours4-6-sorted-embedding-distance-full-ckpt-191
#     # ours4-6-sorted-score-diff-full-ckpt-191
#     ours4-4-sorted-score-diff-full-ckpt-191
#     ours4-4-sorted-score-diff-full
# )

# base_model="mistral-7b"
# loss_type_list=(
#     # dpo-sorted-llama-full
#     # dpo-sorted-llama-full-ckpt-191
#     # dpo-sorted-reward-diff-full
#     # dpo-sorted-score-diff-full
#     # dpo-sorted-embedding-distance-full
#     # ours4-6-sorted-embedding-distance-full
#     ours4-6-sorted-score-diff-full
#     # ours4-8-sorted-reward-diff-full
#     # ours4-4-sorted-score-diff-full
#     ) 

base_model="mistral-7b"
# base_model="llama-3-8b"
loss_type_list=( 
    # ours4-6-sorted-score-diff-full
    # dpo-sorted-score-diff-warmup-full
    # dpo-sorted-score-diff-warmup-full
    # ours4-6-sorted-score-diff-new-base-full
    # dpo-sorted-score-diff-new-base-full
    # dpo-sorted-embedding-distance-new-base-full
    # ours4-6-sorted-score-diff-new-base-full-lr3
    # ours4-6-sorted-score-diff-full-new-base-lr1
    # ours4-6-sorted-score-diff-new-base-full-lr4
    # ours4-6-sorted-score-diff-new-base-full-lr5
    # ours4-6-sorted-score-diff-full-new-base-lr1-ckpt-336
    # ours4-6-sorted-score-diff-new-base-full-lr5-ckpt-336
    # ours4-6-sorted-score-diff-new-base-full-lr6
    # ours4-6-sorted-score-diff-new-base-full-lr7
    # dpo-sorted-score-diff-new-base-full-lr5
    # dpo-sorted-mistral-full
    # selectiveDPO
    # dpo-sorted-mistral-new-base-full
    # ours4-6-sorted-score-diff-new-base-full-lr8
    ours4-6-sorted-score-diff-new-base-full-lr5-replicate

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
        # model_name_or_path="glorgao/SelectiveDPO-Llama3-8B-SFT-UFBinarized"
        if [[ $base_model == "llama-3-8b" ]]; then
            model_name_or_path="glorgao/SelectiveDPO-Llama3-8B-SFT-UFBinarized"
        elif [[ $base_model == "mistral-7b" ]]; then
            model_name_or_path="glorgao/SelectiveDPO-Mistral-7B-SFT-UFBinarized"
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

