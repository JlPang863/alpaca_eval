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

# loss_type_list=("sft" "dpo" "ipo" "cdpo" "robust" "ipo" "dpj")
# loss_type_list=("cdpo-new" "robust-new" "ours1-1") #"dpo-new"  "cdpo-new" "robust-new" "ours1-1" 
# loss_type_list=("dpo-new1"  "cdpo-new1" "robust-new1" "ours1-1-new1" )

# loss_type_list=("robust-new1")
# loss_type_list=("dpo-random" "dpo-top" "dpo-bottom") #

# loss_type_list=("dpo-new1" "cdpo-new1" "robust-new1" "ours1-1-new1")
# loss_type_list=("ours1-2-new1") 
# loss_type_list=("kto") 
# loss_type_list=("ours4-1-new1") 
# loss_type_list=("kto-filtered") 
# loss_type_list=("ours4-2-new1") 

# loss_type_list=("dpo-new1-sorted-llama-full" "dpo-new1-sorted-score-diff-full" "dpo-new1-sorted-llama-half"  "dpo-new1-sorted-score-diff-half") #"dpo-new1-sorted-llama-full" "dpo-new1-sorted-score-diff-full" "dpo-new1-sorted-llama-half"  "dpo-new1-sorted-score-diff-half"
# loss_type_list=("dpo-new1-sorted-llama-half-new-params" "dpo-new1-sorted-score-diff-half-new-params")

# loss_type_list=("dpo-new1-sorted-reward-diff-half")

# loss_type_list=("dpo-random-identical" "dpo-random-identical-reverse") #

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

# loss_type_list=("sft-identical-pairs-7387")
# loss_type_list=("sft-identical-pairs-14774")
# loss_type_list=("dpo-new1-sorted-reward-diff-full")
# loss_type_list=("sft-identical-pairs-7387-rejected")

# loss_type_list=("ours4-2-new1-sorted-score-diff-full-new")
# loss_type_list=("ours4-2-new1-sorted-score-diff-half-new")

# loss_type_list=("ours4-4-new1-sorted-score-diff-full-new" "ours4-4-new1-new-sorted-score-diff-full-new" "ours4-5-new1-sorted-score-diff-full-new")

# loss_type_list=("ours4-4-new1-sorted-score-diff-full-new-ckpt-191" "ours4-4-new1-new-sorted-score-diff-full-new-ckpt-191" "ours4-5-new1-sorted-score-diff-full-new-ckpt-191")

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
#     "ours4-8-sorted-reward-diff-full"
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
    # ours4-6-sorted-score-diff-full
    # ours4-6-new1-new-sorted-score-diff-full-new-ckpt-192
    # ours4-7-new1-new-sorted-score-diff-full-new-ckpt-191
    ours4-6-sorted-score-diff-full-ckpt-96
    ours4-6-sorted-score-diff-full-ckpt-192
    ours4-6-sorted-score-diff-full-ckpt-288
)


# response_file="model_outputs_20.json" ###test
response_file="model_outputs_full.json" ###full


for loss_type in ${loss_type_list[@]}; do

  # model_output_file="model_outputs/${base_model}-${loss_type}-merged/model_outputs_20.json"
    echo "*** loss type: $loss_type ***"

  if [[ $loss_type == "sft" ]]; then
      if [[ $base_model == "llama-3-8b" ]]; then
          model_output_file="model_outputs/Llama-3-Base-8B-SFT/${response_file}"
      elif [[ $base_model == "mistral-7b" ]]; then
          model_output_file="model_outputs/zephyr-7b-sft-full/${response_file}"
      else
          model_output_file="${root_path}/${base_model}-sft/${response_file}"
      fi
  else
    model_output_file="model_outputs/${base_model}-${loss_type}/${response_file}"
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