# chat
```
python -m vllm.entrypoints.openai.api_server \
    --model /home/rickychen/桌面/llm/models/Llama-3.2-11B-Vision-Instruct \
    --tensor-parallel-size 2 \
    --max-model-len 64 \
    --max-num-batched-tokens 64 \
    --block-size 8 \
    --gpu-memory-utilization 0.95 \
    --max-num-seqs 1 \
    --enforce-eager \
    --dtype bfloat16 \
    --port 8001 \
    --disable-log-stats \
    --disable-custom-all-reduce \
    --num-gpu-blocks-override 16 \
    --served-model-name Infinirc-Llama3.2-8B-5G-v1.0
```



```


    --preprocessing_num_workers 16 \
    --finetuning_type lora \
    --template default \
    --flash_attn auto \
    --dataset_dir data \
    --dataset train4 \
    --cutoff_len 32768 \
    --learning_rate 5e-05 \
    --num_train_epochs 5.0 \
    --max_samples 588001 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --lr_scheduler_type cosine \
    --max_grad_norm 1.0 \
    --logging_steps 2 \
    --save_steps 5000 \
    --warmup_steps 0 \
    --optim adamw_torch \
    --packing False \
    --report_to none \
    --output_dir saves/Custom/lora/train_2024-10-29-10-20-21 \
    --bf16 True \
    --plot_loss True \
    --ddp_timeout 180000000 \
    --include_num_input_tokens_seen True \
    --lora_rank 8 \
    --lora_alpha 16 \
    --lora_dropout 0 \
    --lora_target all \
    --val_size 0.03 \
    --eval_strategy steps \
    --eval_steps 5000 \
    --per_device_eval_batch_size 1
```


```
CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server --gpu-memory-utilization 0.95 --model=meta-llama/Llama-3.2-11B-Vision-Instruct --tokenizer=meta-llama/Llama-3.2-11B-Vision-Instruct --download-dir="/home/user/storage/hf_cache" --dtype=bfloat16 --device=cuda --host=127.0.0.1 --port=8080 --max-model-len=8192 --quantization="fp8" --enforce_eager --max_num_seqs=8 --enable_chunked-prefill=false
```
---
16G VRAM

CUDA_VISIBLE_DEVICES=0,1 python -m vllm.entrypoints.openai.api_server --gpu-memory-utilization 0.95 --model=meta-llama/Llama-3.2-11B-Vision-Instruct --tokenizer=meta-llama/Llama-3.2-11B-Vision-Instruct --download-dir="/home/user/storage/hf_cache" --dtype=bfloat16 --device=cuda --host=127.0.0.1 --port=8080 --max-model-len=8192 --quantization="fp8" --enforce_eager --max_num_seqs=8 --enable_chunked-prefill=false --tensor_parallel_size=2



CUDA_VISIBLE_DEVICES=0,1 python -m vllm.entrypoints.openai.api_server --gpu-memory-utilization 0.95 --model=meta-llama/Llama-3.2-11B-Vision-Instruct --tokenizer=meta-llama/Llama-3.2-11B-Vision-Instruct --download-dir="/home/user/storage/hf_cache" --dtype=bfloat16 --device=cuda --host=127.0.0.1 --port=8000 --max-model-len=8192 --enforce_eager --max_num_seqs=8 --enable_chunked-prefill=false --tensor_parallel_size=2


CUDA_VISIBLE_DEVICES=0,1 python -m vllm.entrypoints.openai.api_server --gpu-memory-utilization 0.99 --model=meta-llama/Llama-3.2-11B-Vision-Instruct --tokenizer=meta-llama/Llama-3.2-11B-Vision-Instruct --download-dir="/home/user/storage/hf_cache" --dtype=bfloat16 --device=cuda --host=127.0.0.1 --port=8000 --max-model-len=512 --enforce_eager --max_num_seqs=4 --enable_chunked-prefill=false --tensor_parallel_size=2


CUDA_VISIBLE_DEVICES=0,1 python -m vllm.entrypoints.openai.api_server --gpu-memory-utilization 0.99 --model=meta-llama/Llama-3.2-11B-Vision-Instruct --tokenizer=meta-llama/Llama-3.2-11B-Vision-Instruct --download-dir="/home/user/storage/hf_cache" --dtype=bfloat16 --device=cuda --host=127.0.0.1 --port=8080 --max-model-len=4096 --enforce_eager --max_num_seqs=4 --enable_chunked-prefill=false --tensor_parallel_size=2


CUDA_VISIBLE_DEVICES=0,1 python -m vllm.entrypoints.openai.api_server \
    --gpu-memory-utilization 0.98 \
    --model meta-llama/Llama-3.2-11B-Vision-Instruct \
    --tokenizer meta-llama/Llama-3.2-11B-Vision-Instruct \
    --dtype bfloat16 \
    --device cuda \
    --host 127.0.0.1 \
    --port 8000 \
    --max-model-len 2048 \
    --enforce-eager \
    --max-num-seqs 2 \
    --enable-chunked-prefill false \
    --tensor-parallel-size 2 \
    --trust-remote-code \
    --download-dir /home/rickychen/桌面/llm/models/Llama-3.2-11B-Vision-Instruct \
    --block-size 8 \
    --swap-space 16



---
16G no fp8


CUDA_VISIBLE_DEVICES=0,1 python -m vllm.entrypoints.openai.api_server --gpu-memory-utilization 0.95 --model=meta-llama/Llama-3.2-11B-Vision-Instruct --tokenizer=meta-llama/Llama-3.2-11B-Vision-Instruct --dtype=bfloat16 --device=cuda --host=0.0.0.0 --port=8001 --max-model-len=8000 --enforce_eager --max_num_seqs=4 --enable_chunked-prefill=false --tensor_parallel_size=2

CUDA_VISIBLE_DEVICES=0,1 python -m vllm.entrypoints.openai.api_server --gpu-memory-utilization 0.95 --model=meta-llama/Llama-3.2-11B-Vision-Instruct --tokenizer=meta-llama/Llama-3.2-11B-Vision-Instruct --dtype=bfloat16 --device=cuda --host=0.0.0.0 --port=8001 --max-model-len=8000 --enforce_eager --max_num_seqs=6 --enable_chunked-prefill=false --tensor_parallel_size=2
