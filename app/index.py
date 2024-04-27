# Import necessary modules
from flask import Flask, render_template, request
# from gpt import answer_question
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import cloudpickle
import pickle
from flask import session

# Initialize Flask app
# Initialize Flask app with a secret key
app = Flask(__name__)
app.secret_key = b'_5#y2L"F4Q8z\n\xec]/'
model_directory_triage = "./triage_bert"
# model_directory = "./model_t5"  # Adjust the path to match where you've saved the model
model_directory_relevant = "./relevantQA_bert"

device = "cuda" if torch.cuda.is_available() else "cpu"

# load_model = T5ForConditionalGeneration.from_pretrained(model_directory)
# load_model.to(device)
# load_tokenizer = T5Tokenizer.from_pretrained(model_directory)
# load_model = load_model.eval()

load_model_triage = BertForSequenceClassification.from_pretrained(model_directory_triage)
load_model_triage.to(device)
load_tokenizer_triage = BertTokenizer.from_pretrained(model_directory_triage)
load_model_triage = load_model_triage.eval()

load_model_relevant = BertForSequenceClassification.from_pretrained(model_directory_relevant)
load_model_relevant.to(device)
load_tokenizer_relevant = BertTokenizer.from_pretrained(model_directory_relevant)
load_model_relevant = load_model_relevant.eval()

# Load the label encoder from a file
with open('./label_encoder/relevant_encoder.pkl', 'rb') as f:
    label_encoder_relevant = pickle.load(f)

# Load the label encoder from a file
with open('./label_encoder/traige_encoder.pkl', 'rb') as f:
    label_encoder_traige = pickle.load(f)

max_len = 128

def load_model():
    file_path ="D:/AIT/Sem2/NLP/medicalQA/app/model/medical_chatbot_pickle_version3.pkl"
    with open(file_path, 'rb') as f:
        chain = cloudpickle.load(f)
    return chain

def answer_question(query):
    model = load_model()
    answer = model({"question":query})
    answer_text = answer['answer']
    print (answer_text)
    return answer_text

def predict_text_relevant(model, text, tokenizer, max_len, device):
    model.eval()
    encoding = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=max_len,
        return_token_type_ids=False,
        padding='max_length',
        return_attention_mask=True,
        return_tensors='pt',
        truncation=True,
    )
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs[0]
        prediction = torch.argmax(logits, dim=1).cpu().numpy()[0]

    return prediction


def predict_text(model, text, tokenizer, max_len, device):
    model.eval()
    encoding = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=max_len,
        return_token_type_ids=False,
        padding='max_length',
        return_attention_mask=True,
        return_tensors='pt',
        truncation=True,
    )
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs[0]
        prediction = torch.argmax(logits, dim=1).cpu().numpy()[0]

    return prediction

# Route for the home page
@app.route('/')
def home():
    prompt_message = "Hello! I'm your Medical Assistant, here to provide you with insights and information about health conditions, treatments, and general medical advice. Whether you need information on symptoms, advice on health issues, or details about medical procedures, feel free to ask, and I'll do my best to provide you with clear and informative answers."
    return render_template('pages/index.html', prompt_message=prompt_message)

# Route for handling form submission
@app.route('/qa', methods=['GET', 'POST'])
def qa():
    generated_response = None
    user_message = ""

    # Load chat history if it exists, otherwise initialize an empty list
    chat_history = session.get('chat_history', [])

    if request.method == 'POST':
        user_message = request.form['user_message']
        relevant_label = predict_text_relevant(load_model_relevant, user_message, load_tokenizer_relevant, max_len, device)
        predicted_label_relevant = label_encoder_relevant.inverse_transform([relevant_label])[0]
        if predicted_label_relevant== "relevant":
            predict_text_emergency = predict_text(load_model_triage,user_message, load_tokenizer_triage, max_len,device)
            predicted_label_triage = label_encoder_traige.inverse_transform([predict_text_emergency])[0]
            if (predicted_label_triage=="non-urgent"):
                generated_response = answer_question(user_message)
            else:
                generated_response = "This is a medical emergency so please contact your local hospital or emergency responder. For emergency medical assistance in Thailand, please dial 1669." 
        else :
            generated_response = "This is not a question of relevancy with a medical situation and this application is only able to handle queries related to health and medicine"

        # Append current message and response to chat history
        chat_history.append({'user_message': user_message, 'generated_response': generated_response})

        # Save chat history to session
        session['chat_history'] = chat_history

    return render_template('pages/index.html', generated_response=generated_response, user_message=user_message, chat_history=chat_history)

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
