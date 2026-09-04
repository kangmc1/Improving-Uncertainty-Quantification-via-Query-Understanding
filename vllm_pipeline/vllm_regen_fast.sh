# 결과 관리를 어떻게 할까;;

# 1. llama
# python regen_fast.py --model_name llama3.1-8b-it --data_name triviaqa --format direct --format2 rec --look vote --device_map 0,1
# python regen_fast.py --model_name llama3.1-8b-it --data_name triviaqa --format reflect2 --format2 rec --look vote --device_map 0,1
# python regen_fast.py --model_name llama3.1-8b-it --data_name triviaqa --format direct --format2 rag --look vote --device_map 0,1
# python regen_fast.py --model_name llama3.1-8b-it --data_name triviaqa --format reflect2 --format2 rag --look vote --device_map 0,1
# python regen_fast.py --model_name llama3.1-8b-it --data_name nq --format direct --format2 rec --look vote --device_map 0,1
# python regen_fast.py --model_name llama3.1-8b-it --data_name nq --format reflect2 --format2 rec --look vote --device_map 0,1
# python regen_fast.py --model_name llama3.1-8b-it --data_name nq --format direct --format2 rag --look vote --device_map 0,1
# python regen_fast.py --model_name llama3.1-8b-it --data_name nq --format reflect2 --format2 rag --look vote --device_map 0,1
# python regen_fast.py --model_name llama3.1-8b-it --data_name webq --format direct --format2 rec --look vote --device_map 0,1
# python regen_fast.py --model_name llama3.1-8b-it --data_name webq --format reflect2 --format2 rec --look vote --device_map 0,1
# python regen_fast.py --model_name llama3.1-8b-it --data_name webq --format direct --format2 rag --look vote --device_map 0,1
# python regen_fast.py --model_name llama3.1-8b-it --data_name webq --format reflect2 --format2 rag --look vote --device_map 0,1

# 2. qwen
# python regen_fast.py --model_name qwen2.5-7b-it --data_name triviaqa --format direct --format2 rec --look vote --device_map 0,1
# python regen_fast.py --model_name qwen2.5-7b-it --data_name triviaqa --format reflect2 --format2 rec --look vote --device_map 0,1
# python regen_fast.py --model_name qwen2.5-7b-it --data_name triviaqa --format direct --format2 rag --look vote --device_map 0,1
# python regen_fast.py --model_name qwen2.5-7b-it --data_name triviaqa --format reflect2 --format2 rag --look vote --device_map 0,1
# python regen_fast.py --model_name qwen2.5-7b-it --data_name nq --format direct --format2 rec --look vote --device_map 0,1
# python regen_fast.py --model_name qwen2.5-7b-it --data_name nq --format reflect2 --format2 rec --look vote --device_map 0,1
# python regen_fast.py --model_name qwen2.5-7b-it --data_name nq --format direct --format2 rag --look vote --device_map 0,1
# python regen_fast.py --model_name qwen2.5-7b-it --data_name nq --format reflect2 --format2 rag --look vote --device_map 0,1
# python regen_fast.py --model_name qwen2.5-7b-it --data_name webq --format direct --format2 rec --look vote --device_map 0,1
# python regen_fast.py --model_name qwen2.5-7b-it --data_name webq --format reflect2 --format2 rec --look vote --device_map 0,1
# python regen_fast.py --model_name qwen2.5-7b-it --data_name webq --format direct --format2 rag --look vote --device_map 0,1
# python regen_fast.py --model_name qwen2.5-7b-it --data_name webq --format reflect2 --format2 rag --look vote --device_map 0,1

# 3. mistral(정식명칭은 ministral)
# python regen_fast.py --model_name mistral-8b-it --data_name triviaqa --format direct --format2 rec --look vote --device_map 0,1
# python regen_fast.py --model_name mistral-8b-it --data_name triviaqa --format reflect2 --format2 rec --look vote --device_map 0,1
# python regen_fast.py --model_name mistral-8b-it --data_name triviaqa --format direct --format2 rag --look vote --device_map 0,1
# python regen_fast.py --model_name mistral-8b-it --data_name triviaqa --format reflect2 --format2 rag --look vote --device_map 0,1
# python regen_fast.py --model_name mistral-8b-it --data_name nq --format direct --format2 rec --look vote --device_map 0,1
# python regen_fast.py --model_name mistral-8b-it --data_name nq --format reflect2 --format2 rec --look vote --device_map 0,1
# python regen_fast.py --model_name mistral-8b-it --data_name nq --format direct --format2 rag --look vote --device_map 0,1
# python regen_fast.py --model_name mistral-8b-it --data_name nq --format reflect2 --format2 rag --look vote --device_map 0,1
# python regen_fast.py --model_name mistral-8b-it --data_name webq --format direct --format2 rec --look vote --device_map 0,1
# python regen_fast.py --model_name mistral-8b-it --data_name webq --format reflect2 --format2 rec --look vote --device_map 0,1
# python regen_fast.py --model_name mistral-8b-it --data_name webq --format direct --format2 rag --look vote --device_map 0,1
# python regen_fast.py --model_name mistral-8b-it --data_name webq --format reflect2 --format2 rag --look vote --device_map 0,1


# 일단 rag는 다시 리셋해야됨. [Answer]가 그대로 들어감;
# 5. olmo
# python regen_fast.py --model_name olmo2-7b-it --data_name triviaqa --format direct --format2 rec --look vote --device_map 0,1
# python regen_fast.py --model_name olmo2-7b-it --data_name triviaqa --format reflect2 --format2 rec --look vote --device_map 0,1
# python regen_fast.py --model_name olmo2-7b-it --data_name triviaqa --format direct --format2 rag --look vote --device_map 0,1
# python regen_fast.py --model_name olmo2-7b-it --data_name triviaqa --format reflect2 --format2 rag --look vote --device_map 0,1
# python regen_fast.py --model_name olmo2-7b-it --data_name nq --format direct --format2 rec --look vote --device_map 0,1
# python regen_fast.py --model_name olmo2-7b-it --data_name nq --format reflect2 --format2 rec --look vote --device_map 0,1
# python regen_fast.py --model_name olmo2-7b-it --data_name nq --format direct --format2 rag --look vote --device_map 0,1
# python regen_fast.py --model_name olmo2-7b-it --data_name nq --format reflect2 --format2 rag --look vote --device_map 0,1
# python regen_fast.py --model_name olmo2-7b-it --data_name webq --format direct --format2 rec --look vote --device_map 0,1
# python regen_fast.py --model_name olmo2-7b-it --data_name webq --format reflect2 --format2 rec --look vote --device_map 0,1
# python regen_fast.py --model_name olmo2-7b-it --data_name webq --format direct --format2 rag --look vote --device_map 0,1
# python regen_fast.py --model_name olmo2-7b-it --data_name webq --format reflect2 --format2 rag --look vote --device_map 0,1


# ablation study
# python regen_fast.py --model_name llama3.1-8b-it --data_name nq --format reflect2 --format2 rec --look one --device_map 0,1
# python regen_fast.py --model_name llama3.1-8b-it --data_name nq --format reflect2 --format2 rec --look two --device_map 0,1
# python regen_fast.py --model_name llama3.1-8b-it --data_name nq --format reflect2 --format2 rec --look three --device_map 0,1
# python regen_fast.py --model_name qwen2.5-7b-it --data_name nq --format reflect2 --format2 rec --look one --device_map 0,1
# python regen_fast.py --model_name qwen2.5-7b-it --data_name nq --format reflect2 --format2 rec --look two --device_map 0,1
# python regen_fast.py --model_name qwen2.5-7b-it --data_name nq --format reflect2 --format2 rec --look three --device_map 0,1
# python regen_fast.py --model_name olmo2-7b-it --data_name nq --format reflect2 --format2 rec --look one --device_map 0,1
# python regen_fast.py --model_name olmo2-7b-it --data_name nq --format reflect2 --format2 rec --look two --device_map 0,1
# python regen_fast.py --model_name olmo2-7b-it --data_name nq --format reflect2 --format2 rec --look three --device_map 0,1
# python regen_fast.py --model_name mistral-8b-it --data_name nq --format reflect2 --format2 rec --look one --device_map 0,1
# python regen_fast.py --model_name mistral-8b-it --data_name nq --format reflect2 --format2 rec --look two --device_map 0,1
# python regen_fast.py --model_name mistral-8b-it --data_name nq --format reflect2 --format2 rec --look three --device_map 0,1

python regen_fast.py --model_name llama3.1-8b-it --data_name triviaqa --format reflect2 --format2 rec --look one --device_map 0,1
python regen_fast.py --model_name llama3.1-8b-it --data_name triviaqa --format reflect2 --format2 rec --look two --device_map 0,1
python regen_fast.py --model_name llama3.1-8b-it --data_name triviaqa --format reflect2 --format2 rec --look three --device_map 0,1
python regen_fast.py --model_name qwen2.5-7b-it --data_name triviaqa --format reflect2 --format2 rec --look one --device_map 0,1
python regen_fast.py --model_name qwen2.5-7b-it --data_name triviaqa --format reflect2 --format2 rec --look two --device_map 0,1
python regen_fast.py --model_name qwen2.5-7b-it --data_name triviaqa --format reflect2 --format2 rec --look three --device_map 0,1
python regen_fast.py --model_name olmo2-7b-it --data_name triviaqa --format reflect2 --format2 rec --look one --device_map 0,1
python regen_fast.py --model_name olmo2-7b-it --data_name triviaqa --format reflect2 --format2 rec --look two --device_map 0,1
python regen_fast.py --model_name olmo2-7b-it --data_name triviaqa --format reflect2 --format2 rec --look three --device_map 0,1
python regen_fast.py --model_name mistral-8b-it --data_name triviaqa --format reflect2 --format2 rec --look one --device_map 0,1
python regen_fast.py --model_name mistral-8b-it --data_name triviaqa --format reflect2 --format2 rec --look two --device_map 0,1
python regen_fast.py --model_name mistral-8b-it --data_name triviaqa --format reflect2 --format2 rec --look three --device_map 0,1

python regen_fast.py --model_name llama3.1-8b-it --data_name webq --format reflect2 --format2 rec --look one --device_map 0,1
python regen_fast.py --model_name llama3.1-8b-it --data_name webq --format reflect2 --format2 rec --look two --device_map 0,1
python regen_fast.py --model_name llama3.1-8b-it --data_name webq --format reflect2 --format2 rec --look three --device_map 0,1
python regen_fast.py --model_name qwen2.5-7b-it --data_name webq --format reflect2 --format2 rec --look one --device_map 0,1
python regen_fast.py --model_name qwen2.5-7b-it --data_name webq --format reflect2 --format2 rec --look two --device_map 0,1
python regen_fast.py --model_name qwen2.5-7b-it --data_name webq --format reflect2 --format2 rec --look three --device_map 0,1
python regen_fast.py --model_name olmo2-7b-it --data_name webq --format reflect2 --format2 rec --look one --device_map 0,1
python regen_fast.py --model_name olmo2-7b-it --data_name webq --format reflect2 --format2 rec --look two --device_map 0,1
python regen_fast.py --model_name olmo2-7b-it --data_name webq --format reflect2 --format2 rec --look three --device_map 0,1
python regen_fast.py --model_name mistral-8b-it --data_name webq --format reflect2 --format2 rec --look one --device_map 0,1
python regen_fast.py --model_name mistral-8b-it --data_name webq --format reflect2 --format2 rec --look two --device_map 0,1
python regen_fast.py --model_name mistral-8b-it --data_name webq --format reflect2 --format2 rec --look three --device_map 0,1