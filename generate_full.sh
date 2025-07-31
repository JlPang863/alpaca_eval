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
# loss_type_list=("dpo-new1-sorted-llama-half"  "dpo-new1-sorted-score-diff-half") #"dpo-new1-sorted-llama-full" "dpo-new1-sorted-score-diff-full" "dpo-new1-sorted-llama-half"  "dpo-new1-sorted-score-diff-half"

# loss_type_list=("dpo-new1-sorted-llama-half-new-params")

# loss_type_list=("dpo-new1-sorted-llama-half-new-params" "dpo-new1-sorted-score-diff-half-new-params")

# loss_type_list=("dpo-random-identical" "dpo-new1-sorted-reward-diff-half")

# loss_type_list=("dpo-sorted-score-diff-subset" "dpo-new1-sorted-score-diff-subset-new-params")

# loss_type_list=("ours4-2-new1-new-params")

# loss_type_list=("dpo-random-identical-reward-score-based-swap")

# loss_type_list=("sft-kto-random-identical-subset-lora")
# loss_type_list=("sft-kto-random-identical-subset-chosen-only-lora")
# loss_type_list=("dpo-new1-sorted-reward-diff-swap-half" "dpo-new1-sorted-reward-diff-swap-full" "dpo-new1-sorted-docta-score-diff-full" "dpo-new1-sorted-docta-score-diff-swap-half")

# loss_type_list=("dpo-new1-sorted-docta-score-diff-swap-full" "dpo-new1-sorted-score-diff-full-new-params")


# loss_type_list=("ours4-1-new1-sorted-docta-score-diff-full" "ours4-2-new1-sorted-score-diff-full-new-params")

# loss_type_list=("ours4-1-new1-sorted-docta-score-diff-half" "ours4-2-new1-sorted-score-diff-half-new-params")
# loss_type_list=("dpo-new1-new-sorted-score-diff-full-new-params" "ours4-1-new1-new-sorted-docta-score-diff-full" "ours4-3-new1-sorted-score-diff-full-new-params")


# loss_type_list=("dpo-new1-new-sorted-score-diff-full-new-params-checkpoint-190" "ours4-1-new1-new-sorted-docta-score-diff-full-checkpoint-190" "ours4-3-new1-sorted-score-diff-full-new-params-checkpoint-190")

# loss_type_list=("dpo-new1-new-sorted-score-diff-full-new-params-checkpoint-95" "ours4-1-new1-new-sorted-docta-score-diff-full-checkpoint-95" "dpo-new1-new-sorted-score-diff-full-new-params-checkpoint-285" "ours4-1-new1-new-sorted-docta-score-diff-full-checkpoint-285")

# loss_type_list=("ours4-1-new1-new-sorted-docta-score-diff-full-lr1" "dpo-new1-new-sorted-llama-full-checkpoint-95" "dpo-new1-new-sorted-llama-full")

# loss_type_list=("dpo-new1-new-sorted-llama-full-checkpoint-190" "ours4-1-new1-new-sorted-docta-score-diff-full-lr1-checkpoint-190")


# loss_type_list=("dpo-new1-new-sorted-docta-score-diff-full" "ours4-2-new1-new-sorted-score-diff-full")

# loss_type_list=(
# "dpo-identical-pairs-7387"
# "dpo-sorted-identical-pairs-7387"
# "dpo-reverse-sorted-identical-pairs-7387"
# "dpo-identical-pairs-swap-7387"
# "dpo-identical-pairs-high-quality-1000"
# "dpo-identical-pairs-low-quality-1000")
# loss_type_list=("llama-3-8b-sft-identical-pairs-7387")
# loss_type_list=("sft-identical-pairs-14774")

# loss_type_list=("dpo-new1-sorted-reward-diff-full")
# loss_type_list=("sft-identical-pairs-7387-rejected")
# loss_type_list=("ours4-2-new1-sorted-score-diff-full-new")

# loss_type_list=("ours4-2-new1-sorted-score-diff-half-new")

# loss_type_list=("ours4-4-new1-sorted-score-diff-full-new" "ours4-4-new1-new-sorted-score-diff-full-new" "ours4-5-new1-sorted-score-diff-full-new")



# loss_type_list=("ours4-4-new1-sorted-score-diff-full-new-ckpt-191" "ours4-4-new1-new-sorted-score-diff-full-new-ckpt-191" "ours4-5-new1-sorted-score-diff-full-new-ckpt-191")

# sleep 40m

loss_type_list=(
# "dpo-identical-pairs-7387"
# "dpo-sorted-identical-pairs-7387"
# "dpo-reverse-sorted-identical-pairs-7387"
# "dpo-identical-pairs-swap-7387"
# "dpo-identical-pairs-high-quality-1000"
# "dpo-identical-pairs-low-quality-1000"
# "ours4-4-identical-pairs-7387"
# "ours4-4-sorted-identical-pairs-7387"
# "ours4-4-new1-new-sorted-score-diff-full-new-ckpt-288"
# "ours4-6-new1-new-sorted-score-diff-full-new"
# "ours4-7-new1-new-sorted-score-diff-full-new"
# "dpo-new1-new-sorted-reward-diff-full"
# "ours4-6-new1-new-sorted-score-diff-full-new-ckpt-96"
# "dpo-new1-sorted-score-diff-full-sft-combine"
# "ours4-4-new1-sorted-score-diff-full-sft-combine"
    # "dpo-new1-sorted-score-diff-full-sft-combine-ckpt-168"
    # "ours4-4-new1-sorted-score-diff-full-sft-combine-ckpt-168"
# "ours4-8-sorted-reward-diff-full"
# "ours4-8-sorted-reward-diff-swap-warmup-full"
# "dpo-sorted-reward-diff-swap-warmup-full"
# "ours4-8-sorted-reward-diff-swap-warmup-full-ckpt-96"
# "ours4-8-sorted-reward-diff-swap-warmup-full-ckpt-192"
# "dpo-sorted-reward-diff-swap-warmup-full-ckpt-96"
# "dpo-sorted-reward-diff-swap-warmup-full-ckpt-192"
# ours4-8-sorted-reward-diff-swap-warmup-full-ckpt-288
# dpo-sorted-reward-diff-swap-warmup-full-ckpt-288
    # "dpo-sorted-reward-diff-swap-warmup-full-subset"
    # "ours4-8-sorted-reward-diff-swap-warmup-full-subset"
    # "ours4-8-sorted-reward-diff-swap-warmup-full-shuffle"
    # "ours4-8-sorted-reward-diff-full-shuffle"
    # "spa-sorted-reward-diff-full"
    # "spa-sorted-reward-diff-full-shuffle"
    # dpo-sorted-embedding-distance-full
    # "dpo-identical-pairs-7387-revised"
    # dpo-new1-sorted-llama-reverse-full
    # dpo-sorted-score-diff-full-revised-shuffle ##default 
    # dpo-sorted-score-diff-full-revised
    # ours4-6-sorted-score-diff-full
    # ours4-6-new1-new-sorted-score-diff-full-new-ckpt-192
    # # ours4-7-new1-new-sorted-score-diff-full-new-ckpt-191
    # ours4-6-sorted-score-diff-full-ckpt-96
    # ours4-6-sorted-score-diff-full-ckpt-192
    # ours4-6-sorted-score-diff-full-ckpt-288
    dpo-sorted-llama-full-replicate
)


for loss_type in ${loss_type_list[@]}; do
    # model_name_or_path="/mnt/data1/jinlong/DPO-noisy-outputs/llama-3-1b-sft"

    echo "*** loss type: $loss_type ***"

    ## model path ###
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
    # model_name_or_path="${root_path}/${loss_type}"

    accelerate launch \
        --num_processes $NUM_GPUS \
        --main_process_port $main_process_port \
        --mixed_precision bf16 \
        generate_response.py \
        --model_name_or_path $model_name_or_path \
        --batch_size $BATCH_SIZE \
        --output_path $OUTPUT_PATH \

done

