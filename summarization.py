import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#model_name = "sshleifer/distilbart-cnn-12-6"
#model_name = "t5-small"
model_name = "philschmid/bart-large-cnn-samsum"

# Load tokenizer and model directly
tokenizer = AutoTokenizer.from_pretrained(model_name)
summarization_model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)

conversation_test = """hello sue, how are you? I'm a bit worried, to be honest. who is sophie? Sophie is an old friend, I used to live with her in Madagascar.
do you still see her? No, I don't see her anymore. why? She kicked me out, after I tried to contact her. 
why did she kick you out ? "
She said I was a bad memory, that I was no longer welcome in her life. 
why? what did you do? I... did you have an affair with her husband ? 
I... "
tell me! "
I may have gotten close to him, but I swear it was just friendship. 
what about maguy? I met Maguy by chance, she offered me a job on Maupiti. 
what happen next? She was clear about what she expected from me, 
and I agreed."""

test_sentences = [
    "User : Hello, how are you? "
    "Sue : I'm a bit worried, to be honest.",
    "User : Who is Sophie? ",
    "Sue :Sophie is an old friend; I used to live with her in Madagascar.",
    "User : Do you still see her? ",
    "Sue :No, I don't see her anymore.",
    "User : Why?"
    "Sue :She kicked me out after I tried to contact her.",
    "User : Why did she kick you out? "
    "Sue :She said I was a bad memory and no longer welcome in her life.",
    "User : Why? What did you do? "
    "Sue :I...",
    "User : Did you have an affair with her husband? "
    "Sue :I...",
    "User : Tell me!",
    "Sue :I may have gotten close to him, but I swear it was just friendship.",
    "User : What about Maguy? "
    "Sue :I met Maguy by chance; she offered me a job on Maupiti.",
    "User : What happened next?"
    "Sue :She was clear about what she expected from me, and I agreed."
]


def summarize_sentences(sentences, max_length):
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

s = summarize_sentences(test_sentences,500)
print(s)
