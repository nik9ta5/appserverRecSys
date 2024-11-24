from transformers import AutoTokenizer, Qwen2ForCausalLM


class LanguageModel():
    def __init__(self, model_name = "Vikhrmodels/Vikhr-Qwen-2.5-0.5B-Instruct", device = 'cpu', cache_dir='./hugg_cache'):
        self.model = Qwen2ForCausalLM.from_pretrained(model_name, cache_dir=cache_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        self.device = device
        self.model.to(self.device)
        
        if self.tokenizer.pad_token == self.tokenizer.eos_token:
            self.tokenizer.pad_token = "[PAD]"


    def generate(self, prompt):
        # Токенизация и генерация текста
        #apply_chat_template
        input_ids = self.tokenizer.apply_chat_template(prompt, truncation=True, add_generation_prompt=True, return_tensors="pt")
        # print("INPUT TOKENIZER")
        # print(input_ids)
        attention_mask = (input_ids != self.tokenizer.pad_token_id).long()
        
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)

        output = self.model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_length=8192,
            temperature=0.1,
            num_return_sequences=1,
            no_repeat_ngram_size=2,
            top_k=10,
            top_p=0.95
        )

        generated_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
        return generated_text