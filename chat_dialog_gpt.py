from transformers import AutoModelForCausalLM, AutoTokenizer, Conversation, ConversationalPipeline


def dialog_gpt_answer(user_input, context):
    model_name = "microsoft/DialoGPT-medium"

    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
    model = AutoModelForCausalLM.from_pretrained(model_name)

    conversation_pipeline = ConversationalPipeline(model=model, tokenizer=tokenizer)

    # Start the conversation with the context and then add the user input
    conversation = Conversation(context)
    conversation.add_user_input(user_input)

    result = conversation_pipeline(conversation)

    return result.generated_responses[-1]


#print(dialog_gpt_answer('What is your name?', 'you are Sue'))
