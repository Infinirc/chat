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
