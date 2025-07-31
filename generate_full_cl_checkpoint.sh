# export CUDA_VISIBLE_DEVICES=7
# NUM_GPUS=1

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
NUM_GPUS=8

main_process_port=29517
root_path="/mnt/data1/jinlong/CL_DPO_outputs"

BATCH_SIZE=16


base_model="llama-3-8b"

loss_type_list=(
    # dpo-full-eval
    ours4-6-sorted-score-diff-full-eval
    ) 
# ckpt_list=(38 76 114 152 190 228 266 304 342 382)
# ckpt_list=(76 152 228 304 382)
# ckpt_list=(38 114 190 266 342)
ckpt_list=(266 304 342 382)

for loss_type in ${loss_type_list[@]}; do

    OUTPUT_PATH="./model_outputs_cl_checkpoint_final/${base_model}-${loss_type}/"
    mkdir -p "$OUTPUT_PATH"

    for ckpt in ${ckpt_list[@]}; do
        echo "*** loss type: $loss_type ***"
        echo "Processing checkpoint $ckpt"

        model_name_or_path="${root_path}/${base_model}-${loss_type}/checkpoint-${ckpt}"

        echo "*** model_name_or_path: ${model_name_or_path} ***"

        accelerate launch \
            --num_processes $NUM_GPUS \
            --main_process_port $main_process_port \
            --mixed_precision bf16 \
            generate_response.py \
            --model_name_or_path $model_name_or_path \
            --batch_size $BATCH_SIZE \
            --output_path $OUTPUT_PATH

    done
done

