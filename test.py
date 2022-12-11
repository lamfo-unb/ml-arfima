from transformers import AutoTokenizer, GPT2LMHeadModel
tokenizer = AutoTokenizer.from_pretrained("gpt2")
input_ids = tokenizer("The business model implemented by Haze Brasil follows a future trend for medical cannabis sector. The intermediation between buyers, importers and doctors", return_tensors="pt").input_ids
model = GPT2LMHeadModel.from_pretrained("gpt2")

outputs = model.generate(input_ids, max_length=300, do_sample=True)

print("Output:\n" + 100 * '-')
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
print("" + 100 * '-')