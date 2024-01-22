from transformers import pipeline


def roberta_answer(question, context):
    model_name = "deepset/roberta-base-squad2"

    # a) Get predictions
    nlp = pipeline('question-answering', model=model_name, tokenizer=model_name)
    QA_input = {
        'question': question,
        'context': context
    }

    res = nlp(QA_input)

    return res['answer']
