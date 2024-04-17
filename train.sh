export MODEL_NAME='runwayml/stable-diffusion-v1-5'
export BASE_DATA_DIR='data'
export OUTPUT_DIR="outputs"
export INSTANCE_NAME=$1
export INSTANCE_DIR="$BASE_DATA_DIR/instance_images"
export CLASS_DIR="$BASE_DATA_DIR/class_images"
echo "INSTANCE_DIR: $INSTANCE_DIR"
export OUTPUT_DIR="$OUTPUT_DIR/$INSTANCE_NAME"

accelerate launch diffusers/examples/dreambooth/train_dreambooth.py \
    --pretrained_model_name_or_path $MODEL_NAME \
    --revision "fp16" \
    --instance_data_dir "/home/xxx/SD_Finetuning/data/instance_images" \
    --class_data_dir $CLASS_DIR \
    --instance_prompt "A photo of zhr $INSTANCE_NAME" \
    --class_prompt "A photo of $INSTANCE_NAME" \
    --with_prior_preservation \
    --prior_loss_weight 1.0 \
    --num_class_images 100 \
    --output_dir $OUTPUT_DIR \
    --resolution 512 \
    --train_text_encoder \
    --train_batch_size 2 \
    --sample_batch_size 2 \
    --max_train_steps 800 \
    --gradient_accumulation_steps 1 \
    --gradient_checkpointing \
    --learning_rate 1e-6 \
    --lr_scheduler 'constant' \
    --lr_warmup_steps=0 \
    --use_8bit_adam \
    --validation_prompt "A photo of a zhr $INSTANCE_NAME" \
    --num_validation_images 4 \
    --mixed_precision="fp16" \
    --enable_xformers_memory_efficient_attention \
    --set_grads_to_none \