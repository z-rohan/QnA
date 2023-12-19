from transformers import BertForQuestionAnswering, BertTokenizer
import torch

class AnsweringModel():
    def __init__(self):
        self.model = BertForQuestionAnswering.from_pretrained('model')
        self.tokenizer = BertTokenizer.from_pretrained('tokenizer')

    # def answer_question(self, question, context):
    #     # Tokenize input question and context
    #     inputs = self.tokenizer.encode_plus(question, context, return_tensors="pt", add_special_tokens=True)
        
    #     # Handle inputs longer than maximum sequence length
    #     max_seq_length = self.tokenizer.model_max_length
    #     stride = 128  # Adjust as needed
    #     input_ids = inputs['input_ids']
    #     token_type_ids = inputs['token_type_ids']

    #     total_length = input_ids.size(1)
    #     start, end = 0, min(max_seq_length, total_length)
    #     answers = []

    #     while start < total_length:
    #         inputs_chunk = {
    #             'input_ids': input_ids[:, start:end],
    #             'token_type_ids': token_type_ids[:, start:end]
    #         }

    #         # Perform inference on the chunk
    #         with torch.no_grad():
    #             outputs = self.model(**inputs_chunk)

    #         start_scores = outputs.start_logits
    #         end_scores = outputs.end_logits

    #         # Get the maximum start and end scores
    #         start_confidence = float(torch.max(torch.softmax(start_scores, dim=1)))
    #         end_confidence = float(torch.max(torch.softmax(end_scores, dim=1)))

    #         answer_start_index = outputs.start_logits.argmax()
    #         answer_end_index = outputs.end_logits.argmax()

    #         predict_answer_tokens = inputs_chunk['input_ids'][0, answer_start_index: answer_end_index + 1]
    #         answer = self.tokenizer.decode(predict_answer_tokens, skip_special_tokens=True)
    #         answers.append((answer, (start_confidence + end_confidence) / 2))

    #         # Update start and end for the next chunk
    #         start = end - stride
    #         end = min(start + max_seq_length, total_length)

    #     # Choose the answer with the highest confidence
    #     best_answer = max(answers, key=lambda x: x[1])
    #     return best_answer

    def answer_question(self, question, content):
        answers = []
        for i in range(len(content.split(' '))//200+1):
            text = ' '.join(content.split(" ")[200*i:200*(i+1)])
            print(">>>>>",i,"<<<<<",text)
            inputs = self.tokenizer.encode_plus(question, text, return_tensors="pt", add_special_tokens=True)
            print("=======>",len(inputs.get('input_ids')[0]))
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            start_scores = outputs.start_logits
            end_scores = outputs.end_logits

            # Get the maximum start and end scores
            start_confidence = float(torch.max(torch.softmax(start_scores, dim=1)))
            end_confidence = float(torch.max(torch.softmax(end_scores, dim=1)))

            answer_start_index = outputs.start_logits.argmax()
            answer_end_index = outputs.end_logits.argmax()

            predict_answer_tokens = inputs.input_ids[0, answer_start_index : answer_end_index + 1]
            answer = self.tokenizer.decode(predict_answer_tokens, skip_special_tokens=True)

            answers.append((answer, (start_confidence + end_confidence)/2))
        
        best_answer = max(answers, key=lambda x: x[1])
        print("LLLLLLLL", best_answer)
        return best_answer