import torch
from transformers import AutoModelWithLMHead, AutoModelForSequenceClassification, AutoTokenizer
from params import Params

class AI:
    def __init__(self):
        self._generator = AutoModelWithLMHead.from_pretrained('/checkpoint-4500')
        self._generator_tokenizer = AutoTokenizer.from_pretrained('tinkoff-ai/ruDialoGPT-medium')    

        self._classifier = AutoModelForSequenceClassification.from_pretrained('tinkoff-ai/response-quality-classifier-base')
        self._classifier_tokenizer = AutoTokenizer.from_pretrained('tinkoff-ai/response-quality-classifier-base')

    def answer(self, dialog, params):
        text = self._prepare_text(dialog)
        answer = self._generate_answers(text, params)
        best_answer = self._choose_best_answer(answer)
        return best_answer.split('@@ВТОРОЙ@@')[0]

    def _prepare_text(self, dialog):
        text = ''
        for i, replic in enumerate(dialog):
            if i % 2 == 0:
                text += ' @@ПЕРВЫЙ@@ ' + replic.strip()
            else:
                text += ' @@ВТОРОЙ@@ ' + replic.strip()
 
        if i % 2 == 0:
            text += ' @@ПЕРВЫЙ@@'
        else:
            text += ' @@ВТОРОЙ@@'
        return text.strip()

    def _generate_answers(self, text: str, params):
        tokenized_text = self._generator_tokenizer(text, return_tensors='pt')
        input_size = tokenized_text.input_ids.size(1)

        answers = self._generator.generate(
            **tokenized_text, 
            max_new_tokens=params.max_new_tokens,
            repetition_penalty=params.repetition_penalty,
            do_sample=params.do_sample,
            top_k=params.top_k,
            top_p=params.top_p,
            temperature=params.temperature,
            num_beams=params.num_beams,
            no_repeat_ngram_size=params.no_repeat_ngram_size,
            length_penalty=params.length_penalty,
            num_return_sequences=params.num_return_sequences,
        )

        answers = answers[:, input_size:]
        answers = list(map(self._generator_tokenizer.decode, answers))
        return answers

    def _choose_best_answer(self, answers):
        tokenized_answers = self._classifier_tokenizer(answers, return_tensors='pt', padding='longest')
        logits = self._classifier(**tokenized_answers).logits
        scores = torch.sigmoid(logits).cpu().detach().numpy().mean(axis=1)
        scores_argmax = scores.argmax(axis=0)
        return answers[scores_argmax.item()]
    