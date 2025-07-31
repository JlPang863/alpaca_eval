export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 

# eval_model="alpaca_eval_gpt4_turbo_fn"
# eval_model="alpaca_eval_gpt4o_mini"
# eval_model="alpaca_eval_gpt4"
# eval_model="alpaca_eval_gpt4_turbo_fn_new"
# eval_model="oasst_pythia_12b"
# eval_model="alpaca_eval_gpt4_fn"

# eval_model="weighted_alpaca_eval_gpt-4o-mini-2024-07-18"
# eval_model="Self-taught-llama3.1-70B-dpo"
# eval_model="alpaca_eval_cot_gpt4_turbo_fn"
# eval_model="alpaca_eval_clf_gpt4_turbo"
# eval_model="alpaca_eval_gpt_o1"

# eval_model="gpt35_turbo_instruct" ## default use now
eval_model="alpaca_eval_gpt4.1"

######################################################################################################
######################################################################################################

# base_model="llama-3-1b" 
# base_model="mistral-7b" 
base_model="llama-3-8b" 

# loss_type_list=(sft orpo rdpo slic-hf dpo cpo ipo simpo kto) #sft
# loss_type_list=(orpo rdpo slic-hf)
# loss_type_list=(
#     # ours4-6-sorted-score-diff-full-filter-out-similar-samples-ckpt-336
#     # ours4-6-sorted-score-diff-full-filter-out-similar-samples
#     # ours4-6-sorted-score-diff-full-shuffle
#     # ours4-6-sorted-score-diff-full-lr1
#     # ours4-6-sorted-llama-full
#     # spa-sorted-reward-diff-full
#     # ours4-1-new1-new-sorted-docta-score-diff-full
#     # ours4-1-new1-sorted-docta-score-diff-full
#     # ours4-8-sorted-reward-diff-swap-warmup-full-ckpt-96
#     # ours4-4-new1-sorted-score-diff-full-sft-combine
#     ours4-6-sorted-score-diff-full-lr1-ckpt-336
#     )
# loss_type_list=(
#     # "dpo-identical-pairs-7387"
#     # "dpo-sorted-identical-pairs-7387"
#     # "dpo-reverse-sorted-identical-pairs-7387"
#     # "dpo-identical-pairs-high-quality-1000"
#     # "dpo-identical-pairs-low-quality-1000"
#     # sft-identical-pairs-7387-rejected
#     # sft-identical-pairs-7387
#     ours4-4-identical-pairs-7387
#     ours4-4-sorted-identical-pairs-7387
# )

# loss_type_list=(
#     # dpo-new1
#     # cdpo-new1
#     # robust-new1
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

# base_model="mistral-7b"
base_model="llama-3-8b"
loss_type_list=( 
    # ours4-6-sorted-score-diff-full
    # dpo-sorted-score-diff-warmup-full
    # dpo-sorted-score-diff-warmup-full
    # ours4-6-sorted-score-diff-new-base-full
    # ours4-6-sorted-score-diff-full-lr1
    # ours4-6-sorted-score-diff-full-lr2
    # ours4-6-sorted-score-diff-full-lr3
    # ours4-6-sorted-score-diff-full-shuffle
    # selectiveDPO
    )


# base_model="qwen-2.5-7b"
# loss_type_list=(
#     # sft
#     # selectiveDPO
#     dpo-full
#     # dpo-sorted-qwen-full
#     ours4-6-sorted-score-diff-full
#     simpo-full
# )

base_model="llama-3-8b"
loss_type_list=( 
    # ours4-6-sorted-score-diff-full
    # dpo-sorted-score-diff-warmup-full
    # dpo-sorted-score-diff-warmup-full
    # ours4-6-sorted-score-diff-new-base-full
    # ours4-6-sorted-score-diff-new-base-full-lr2
    # ours4-6-sorted-score-diff-full-lr1
    # ours4-6-sorted-score-diff-full-lr2
    # ours4-6-sorted-score-diff-full-lr3
    # ours4-6-sorted-score-diff-full-shuffle
    # selectiveDPO
    # dpo-sorted-score-diff-easy-5k-full
    # dpo-sorted-score-diff-middle-5k-full
    # dpo-sorted-score-diff-difficult-5k-full
    # dpo-sorted-score-diff-easy-5k-full-lr1
    # dpo-sorted-score-diff-middle-5k-full-lr1
    # dpo-sorted-score-diff-difficult-5k-full-lr1
    # agrilla-dpo-full
    # agrilla-dpo-sorted-llama-full
    # agrilla-simpo-full
    # agrilla-ours4-6-sorted-score-diff-full
    # agrilla-ours4-6-sorted-score-diff-full-lr1
    # agrilla-ours4-6-sorted-score-diff-full-lr2
    # agrilla-ours4-6-sorted-score-diff-full-lr3
    # agrilla-dpo-sorted-llama-full-lr1
    # agrilla-dpo-sorted-llama-full-lr2
    # agrilla-dpo-full-lr1
    # agrilla-dpo-full-lr2
    # agrilla-dpo-full-lr3
    # agrilla-dpo-full-lr4
    # agrilla-dpo-full-lr5
    # ours4-6-sorted-llama-full
    # agrilla-ours4-6-sorted-score-diff-full-lr4
    # agrilla-ours4-6-sorted-score-diff-full-lr5
    # ipo-sorted-score-diff-full
    # agrilla-simpo-full-lr1
    # agrilla-dpo-sorted-llama-full-lr5
    # simpo-sorted-score-diff-full
    # simpo-sorted-score-diff-full-ckpt-336
    # ipo-sorted-score-diff-full-ckpt-336
    # ours4-6-sorted-llama-full-new
    # ours4-6-sorted-llama-full-new-ckpt-336
    # dpo-sorted-llama-full-replicate
    # dpo-sorted-llama-full-replicate-ckpt-191
    # simpo
    # ours4-6-sorted-score-diff-full
    # ours4-6-sorted-score-diff-full-replicate
    # dpo-sorted-llama-full-replicate1 ###change beta from 0.1 to 0.01
    # dpo-sorted-llama-full-replicate1-ckpt-191
    # ours4-6-sorted-score-diff-full-threshold1
    # ours4-6-sorted-score-diff-full-threshold2
    dpop-full
    )


model_output_path="model_outputs_cl"

# response_file="model_outputs_20.json" ###test
response_file="model_outputs_full.json" ###full


for loss_type in ${loss_type_list[@]}; do

    # model_output_file="model_outputs/${base_model}-${loss_type}-merged/model_outputs_20.json"
    echo "*** loss type: $loss_type ***"
    echo "*** base model: $base_model ***"

    if [[ $base_model == "llama-3-8b" ]]; then
        base_model_name='Llama-3-Base-8B-SFT'
    elif [[ $base_model == "mistral-7b" ]]; then
        base_model_name='Mistral-7B-Base-SFT'
    fi

    if [[ $loss_type == "sft" ]]; then
        if [[ $base_model == "llama-3-8b" ]]; then
            model_output_file="${model_output_path}/Llama-3-Base-8B-SFT/${response_file}"
        elif [[ $base_model == "mistral-7b" ]]; then
            # model_output_file="${model_output_path}/zephyr-7b-sft-full/${response_file}"
            model_output_file="${model_output_path}/mistral-7b-sft-beta/${response_file}"
        elif [[ $base_model == "qwen-2.5-7b" ]]; then
            model_output_file="${model_output_path}/Qwen2.5-7B-sft-ultrachat/${response_file}"
        else
            model_output_file="${root_path}/${base_model}-sft/${response_file}"
        fi
    elif [[ $loss_type == "simpo" ]]; then
        model_output_file="${model_output_path}/${base_model_name}-SimPO/${response_file}"
    elif [[ $loss_type == "dpo" ]]; then
        model_output_file="${model_output_path}/${base_model_name}-DPO/${response_file}"
    elif [[ $loss_type == "simpo" ]]; then
        model_output_file="${model_output_path}${base_model_name}-SimPO/${response_file}"
    elif [[ $loss_type == "ipo" ]]; then
        model_output_file="${model_output_path}/${base_model_name}-IPO/${response_file}"
    elif [[ $loss_type == "kto" ]]; then
        model_output_file="${model_output_path}/${base_model_name}-KTO/${response_file}"
    elif [[ $loss_type == "cpo" ]]; then
        model_output_file="${model_output_path}/${base_model_name}-CPO/${response_file}"
    elif [[ $loss_type == "rdpo" ]]; then
        model_output_file="${model_output_path}/${base_model_name}-RDPO/${response_file}"
    elif [[ $loss_type == "orpo" ]]; then
        model_output_file="${model_output_path}/${base_model_name}-ORPO/${response_file}"
    elif [[ $loss_type == "slic-hf" ]]; then
        model_output_file="${model_output_path}/${base_model_name}-SLiC-HF/${response_file}"
    else
        model_output_file="${model_output_path}/${base_model}-${loss_type}/${response_file}"
    fi

    echo "*** current model output file: ${model_output_file} ***"

    alpaca_eval --model_outputs  $model_output_file \
        --annotators_config $eval_model

done


################################################################################################
################################################################################################
# model_config="/home/jlpang/alpaca_eval/model_configs/configs/llama-3-1b-ours-merged.yaml"
# model_config="/home/jlpang/alpaca_eval/src/alpaca_eval/models_configs/Llama-3-Instruct-8B-SimPO/configs.yaml"


# # need a GPU for local models
# alpaca_eval evaluate_from_model \
#   --model_configs $model_config \
#   --annotators_config $eval_model