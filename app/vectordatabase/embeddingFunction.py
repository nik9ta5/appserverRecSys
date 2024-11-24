import torch
import torch.nn.functional as F

from transformers import AutoTokenizer, AutoModel

from .textPreprocessing import preproccess_text2emb


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask


class EmbeddingFunction:
    def __init__(self, model_name = "ai-forever/sbert_large_nlu_ru", device = 'cpu', cache_dir='./hugg_cache'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir='./hugg_cache')
        self.model = AutoModel.from_pretrained(model_name, cache_dir='./hugg_cache')
        self.device = device
        self.model.to(self.device) #Перенос модели на устройство


    def __call__(self, input):
        inp = preproccess_text2emb(input) #предобработка текста
        inp = self.tokenizer(inp, return_tensors='pt', padding=True, truncation=True, max_length=512)
        inp.to(device=self.device) #Переносим тензор на устройстов
       
        with torch.no_grad():
            outputs = self.model(**inp)
        
        sentence_embeddings = mean_pooling(outputs, inp['attention_mask'])

        #Перенос эмбеддинга на cpu
        embeddings_cpu = sentence_embeddings.cpu()[0].tolist()
        return embeddings_cpu
    

    def embed_documents(self, documents):
        embeddings = [self(doc) for doc in documents]
        return embeddings
    

    def embed_query(self, query: str):
        return self(query)