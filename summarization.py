import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# qa_model_name = "sshleifer/distilbart-cnn-12-6"
# qa_model_name = "t5-small"
model_name = "philschmid/bart-large-cnn-samsum"

# Load tokenizer and model directly
tokenizer = AutoTokenizer.from_pretrained(model_name)
summarization_model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)


def summarize_conversation(sentences, max_length):
    conversation_text = ' '.join([f"\n{s}" for s in sentences])
    inputs = tokenizer(
        conversation_text,
        return_tensors="pt",
        truncation=True,
        max_length=1024  # Adjust based on the model's max input length
    ).to(device)

    with torch.no_grad():
        summary_ids = summarization_model.generate(
            inputs["input_ids"],
            max_length=max_length,
            min_length=30,
            num_beams=2,
            length_penalty=2.0,
            no_repeat_ngram_size=3,
            early_stopping=True
        )
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary


# sentences = ["Who is Bruce ?",
#            "Bruce is a broken man, consumed by his vices. He prefers solitude, and honestly, he does nothing to make himself likable.",
#            "Was he in the Italian Army?",
#              "Yes"
#              ]
#
# s = summarize_conversation(sentences, 500)
# print(s)