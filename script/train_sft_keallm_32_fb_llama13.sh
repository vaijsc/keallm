# CUDA_VISIBLE_DEVICES=0 torchrun --nnodes 1 --node_rank 0 --nproc_per_node 1  --master_addr 127.0.0.1 --master_port 23956\
#                     src/train.py --output_dir ./save/sft/fb15k237/keallm/keallm_32_llama13 \
#                     --stage sft \
#                     --hop 1-hop \
#                     --model_name_or_path meta-llama/Llama-2-13b-chat-hf \
#                     --language_model_path meta-llama/Llama-2-13b-chat-hf \
#                     --kge_model_path ledong0110/FB15k-237-KGE-Roberta-Base \
#                     --model_type keallm \
#                     --template llama2_keallm \
#                     --num_query_tokens 32 \
#                     --train_from_scratch true \
#                     --num_train_epochs 3 \
#                     --save_total_limit 3 \
#                     --load_best_model_at_end true\
#                     --eval_strategy steps \
#                     --save_strategy steps \
#                     --save_steps 20000 \
#                     --eval_steps 20000 \
#                     --logging_first_step true \
#                     --logging_steps 20 \
#                     --bf16 true \
#                     --do_train true \
#                     --do_eval true\
#                     --predict_with_generate false \
#                     --do_predict false \
#                     --top_k 1 \
#                     --max_new_tokens 32 \
#                     --learning_rate 1.0e-4 \
#                     --warmup_ratio 0.1 \
#                     --lr_scheduler_type cosine \
#                     --eval_dataset FB15k-237_roberta \
#                     --ignore_pad_token_for_loss true \
#                     --per_device_eval_batch_size 2 \
#                     --per_device_train_batch_size 1 \
#                     --gradient_accumulation_steps 2\
#                     --dataset FB15k-237_roberta \
#                     --tokenized_path ./tokenized_data/FB15k-237 \
#                     --deepspeed ./ds2.json
                    # --resume_from_checkpoint true\

CUDA_VISIBLE_DEVICES=0 torchrun --nnodes 1 --node_rank 0 --nproc_per_node 1  --master_addr 127.0.0.1 --master_port 23956\
                    src/train.py --output_dir ./save/sft/fb15k237/keallm/keallm_32_llama13 \
                    --stage sft \
                    --hop 1-hop \
                    --model_name_or_path  meta-llama/Llama-2-13b-chat-hf \
                    --language_model_path meta-llama/Llama-2-13b-chat-hf \
                    --kge_model_path ledong0110/FB15k-237-KGE-Roberta-Base \
                    --model_type keallm \
                    --template llama2_keallm \
                    --num_query_tokens 32 \
                    --train_from_scratch true \
                    --num_train_epochs 3 \
                    --save_total_limit 3 \
                    --load_best_model_at_end true\
                    --eval_strategy steps \
                    --save_strategy steps \
                    --save_steps 20000 \
                    --eval_steps 20000 \
                    --logging_first_step true \
                    --logging_steps 20 \
                    --bf16 true \
                    --do_train true \
                    --do_eval false\
                    --predict_with_generate true \
                    --do_predict true \
                    --top_k 1 \
                    --max_new_tokens 32 \
                    --learning_rate 1.0e-4 \
                    --warmup_ratio 0.1 \
                    --lr_scheduler_type cosine \
                    --eval_dataset FB15k-237_roberta \
                    --ignore_pad_token_for_loss true \
                    --per_device_eval_batch_size 2 \
                    --per_device_train_batch_size 1 \
                    --gradient_accumulation_steps 2\
                    --dataset FB15k-237_roberta \
                    --tokenized_path ./tokenized_data/FB15k-237 \
                    --deepspeed ./ds2.json