from transformers import AutoModelForCausalLM, AutoTokenizer

checkpoint = "./sandbox1"

tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForCausalLM.from_pretrained(checkpoint)

batch_encoding = tokenizer("My name is Erfang", return_tensors='pt')

outputs = model.generate(batch_encoding["input_ids"])
print(tokenizer.decode(outputs.squeeze()))
