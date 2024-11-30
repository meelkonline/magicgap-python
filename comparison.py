from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

from api_requests import CompareRequest

# Set device
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Load model and tokenizer
model_name = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)


# Comparison function
def compare(request: CompareRequest):
    # Extract IDs and hypotheses
    ids, hypotheses = zip(*request.hypotheses)  # Unpack IDs and strings

    # Duplicate the premise to match the number of hypotheses
    premises = [request.premise] * len(hypotheses)

    # Tokenize the input batch
    inputs = tokenizer(premises, hypotheses, truncation=True, padding=True, return_tensors="pt").to(device)

    # Perform inference
    outputs = model(inputs["input_ids"], attention_mask=inputs["attention_mask"])

    # Process predictions for the batch
    predictions = torch.softmax(outputs.logits, dim=-1).tolist()
    label_names = ["entailment", "neutral", "contradiction"]

    # Check each hypothesis for entailment score >= min_score
    for hypothesis_id, hypothesis, prediction in zip(ids, hypotheses, predictions):
        prediction_dict = {name: round(float(pred) * 100, 1) for pred, name in zip(prediction, label_names)}
        entailment_score = prediction_dict["entailment"]
        print(f"Premise: {request.premise}")
        print(f"Hypothesis: {hypothesis}")
        print(f"Prediction: {prediction_dict}\n")

        if entailment_score >= request.min_score:
            return hypothesis_id  # Return the ID of the matching hypothesis

    # If no hypothesis meets the threshold, return 'none'
    return "none"


#Example : Define the single premise and multiple hypotheses
#
# premise = """Bruce and Roy, yes. They vanished around midnight, but they reappeared an hour later."""
#
# #premise = "Bruce and Roy, yes. They vanished around midnight, but they reappeared an hour later."
# hypotheses = [
#     [1, "Bruce and Roy disappeared at midnight on maguy's birthday."],
#     [2, "Bruce and Roy disappeared at midnight during maguy's party, and reappeared an hour later."],
#     [3, "Bruce mentioned that he joined the military during World War II."],
#     [4, "Maupiti is a peaceful island without Italians."],
# ]
#
# req = CompareRequest(premise=premise, hypotheses=hypotheses, min_score=90)
# id = compare(req)
# print(id)
