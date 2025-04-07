from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load Mistral-7B instruct model
model_name = "moussaKam/mbarthez-dialogue-summarization"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

login("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")

def convert_qa_to_affirmation(conversation, max_length=200):
    """
    Converts Q/A into affirmations using a French LLM.
    """
    conversation = ' '.join([f"\n{s}" for s in conversation])
    prompt = f"{conversation}"

    inputs = tokenizer(prompt, return_tensors="pt", max_length=2048, truncation=True)

    output_ids = model.generate(
        inputs["input_ids"],
        max_length=max_length,
        num_beams=5,
        length_penalty=1.2,
        no_repeat_ngram_size=2,
        early_stopping=True
    )

    return tokenizer.decode(output_ids[0], skip_special_tokens=True)

# Example Q/A conversation in French
qa_fr = [
    ("Qui est Bruce ?", "Bruce est un homme brisé, rongé par ses vices."),
    ("Était-il dans l'armée italienne?", "Oui, il y a quelques années."),

]

# conversation = """*person1* : Qui est Bruce ?
# *person2* : Bruce est un homme brisé, rongé par ses vices.
# *person1* : Était-il dans l'armée italienne?
# *person2* : Oui, il y a quelques années.
# *person1* : Que fait-il maintenant?
# *person2* : Il est devenu pêcheur et navigue dans l'océan Indien.
# *person1* : Comment s'appelle son bateau?
# *person2* : Son bateau s'appelle le Bamboo."""
#
# # Convert Q/A to affirmations
# affirmations_fr = convert_qa_to_affirmation(qa_fr, 200)
# print("📌 Affirmations en français:\n", affirmations_fr)
