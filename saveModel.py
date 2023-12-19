from transformers import BertForQuestionAnswering, BertTokenizer
import torch

# Load pre-trained BERT model and tokenizer
model_name = 'bert-large-uncased-whole-word-masking-finetuned-squad'  
# model_name = 'allenai/longformer-base-4096'# Pre-trained BERT model fine-tuned for QA
model = BertForQuestionAnswering.from_pretrained(model_name)
tokenizer = BertTokenizer.from_pretrained(model_name)

model.save_pretrained('model')
tokenizer.save_pretrained('tokenizer')




