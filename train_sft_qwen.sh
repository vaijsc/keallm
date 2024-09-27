CUDA_VISIBLE_DEVICES=0,1,3,5 python src/train.py --output_dir ./save/sft/keallm \
                    --stage sft \
                    --model_name_or_path ./KEALLM-Qwen2-Roberta-1.5B \
                    --template qwen \
                    --num_train_epochs 5 \
                    --save_total_limit 3 \
                    --load_best_model_at_end true\
                    --eval_strategy steps \
                    --save_strategy steps \
                    --save_steps 2000 \
                    --eval_steps 2000 \
                    --logging_first_step true \
                    --logging_steps 20 \
                    --do_train true \
                    --do_eval true\
                    --learning_rate 1.0e-4 \
                    --warmup_ratio 0.1 \
                    --lr_scheduler_type cosine \
                    --eval_dataset FB15k-237_roberta \
                    --ignore_pad_token_for_loss true \
                    --per_device_eval_batch_size 3 \
                    --per_device_train_batch_size 1 \
                    --gradient_accumulation_steps 2\
                    --dataset FB15k-237_roberta \
                    --tokenized_path ./qwen2_tokenized_path \
                    # --deepspeed ./ds2.json
                    # --resume_from_checkpoint true\