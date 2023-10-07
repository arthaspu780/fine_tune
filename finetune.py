
#!wget https://storage.googleapis.com/nlprefresh-public/tydiqa_data.zip
#!unzip tydiqa_data
from sklearn.metrics import f1_score
from datasets import load_from_disk
from transformers import AutoTokenizer
from transformers import AutoModelForQuestionAnswering
path = '/content/tydiqa_data/'


tydiqa_data = load_from_disk(path)

idx = 600
start_index = tydiqa_data['train'][idx]['annotations']['minimal_answers_start_byte'][0]
end_index = tydiqa_data['train'][idx]['annotations']['minimal_answers_end_byte'][0]
tydiqa_data['train'][0]['annotations']
flattened_train_data = tydiqa_data['train'].flatten()
flattened_test_data =  tydiqa_data['validation'].flatten()
flattened_train_data = flattened_train_data.select(range(3000))
flattened_test_data = flattened_test_data.select(range(1000))

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-cased-distilled-squad")#模型的选择处
# Processing samples using the 3 steps described.
def process_samples(sample):
    tokenized_data = tokenizer(sample['document_plaintext'], sample['question_text'], truncation="only_first", padding="max_length")

    input_ids = tokenized_data["input_ids"]

    # We will label impossible answers with the index of the CLS token.
    cls_index = input_ids.index(tokenizer.cls_token_id)

    # If no answers are given, set the cls_index as answer.
    if sample["annotations.minimal_answers_start_byte"][0] == -1:
        start_position = cls_index
        end_position = cls_index
    else:
        # Start/end character index of the answer in the text.
        gold_text = sample["document_plaintext"][sample['annotations.minimal_answers_start_byte'][0]:sample['annotations.minimal_answers_end_byte'][0]]
        start_char = sample["annotations.minimal_answers_start_byte"][0]
        end_char = sample['annotations.minimal_answers_end_byte'][0] #start_char + len(gold_text)

        # sometimes answers are off by a character or two – fix this
        if sample['document_plaintext'][start_char-1:end_char-1] == gold_text:
            start_char = start_char - 1
            end_char = end_char - 1     # When the gold label is off by one character
        elif sample['document_plaintext'][start_char-2:end_char-2] == gold_text:
            start_char = start_char - 2
            end_char = end_char - 2     # When the gold label is off by two characters

        start_token = tokenized_data.char_to_token(start_char)
        end_token = tokenized_data.char_to_token(end_char - 1)

        # if start position is None, the answer passage has been truncated
        if start_token is None:
            start_token = tokenizer.model_max_length
        if end_token is None:
            end_token = tokenizer.model_max_length

        start_position = start_token
        end_position = end_token

    return {'input_ids': tokenized_data['input_ids'],
          'attention_mask': tokenized_data['attention_mask'],
          'start_positions': start_position,
          'end_positions': end_position}

processed_train_data = flattened_train_data.map(process_samples)
processed_test_data = flattened_test_data.map(process_samples)

model = AutoModelForQuestionAnswering.from_pretrained("distilbert-base-cased-distilled-squad")
columns_to_return = ['input_ids','attention_mask', 'start_positions', 'end_positions']
processed_train_data.set_format(type='pt', columns=columns_to_return)
processed_test_data.set_format(type='pt', columns=columns_to_return)


def compute_f1_metrics(pred):
    start_labels = pred.label_ids[0]
    start_preds = pred.predictions[0].argmax(-1)
    end_labels = pred.label_ids[1]
    end_preds = pred.predictions[1].argmax(-1)

    f1_start = f1_score(start_labels, start_preds, average='macro')
    f1_end = f1_score(end_labels, end_preds, average='macro')

    return {
        'f1_start': f1_start,
        'f1_end': f1_end,
    }

from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir='model_results5',
    overwrite_output_dir=True,
    num_train_epochs=1,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=20,
    weight_decay=0.01,
    logging_dir=None,
    logging_steps=50
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=processed_train_data, # training dataset
    eval_dataset=processed_test_data, # evaluation dataset
    compute_metrics=compute_f1_metrics
)

trainer.train()