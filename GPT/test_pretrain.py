from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("/home/zoezhao/scratch/tmp/gpt-xl-ttfs/checkpoint-1000")
print(type(model))
