import torch
import spacy
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# qa_model_name = "sshleifer/distilbart-cnn-12-6"
# qa_model_name = "t5-small"
model_name = "philschmid/bart-large-cnn-samsum"

# Load tokenizer and model directly
tokenizer = AutoTokenizer.from_pretrained(model_name)
summarization_model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)

# Load spaCy's English model for sentence segmentation
nlp = spacy.load("en_core_web_sm")


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
    # Use spaCy to split the summary into sentences
    doc = nlp(summary)
    summary_sentences = [sent.text.strip() for sent in doc.sents]

    return summary_sentences

#
# sentences = ["Who is Bruce ?",
#              "Bruce is a broken man, consumed by his vices. He prefers solitude, and honestly, he does nothing to make himself likable.",
#              "Was he in the Italian Army?",
#              "Yes"
#              ]
#
# sentences2 = ["Qui est Bruce ?",
#               "Bruce est un homme brisé, rongé par ses vices. Il préfère la solitude, et honnêtement, il ne fait rien pour se rendre aimable.",
#               "Etait-il dans l'armée italienne?",
#               "Oui, il y a quelques années"
#               ]
#
# s = summarize_conversation(sentences2, 500)
# print(s)
