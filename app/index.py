# Import necessary modules
from flask import Flask, render_template, request
# from gpt import answer_question
from transformers import T5Tokenizer, T5ForConditionalGeneration, BertTokenizer, BertForSequenceClassification
import torch
import pickle

# Initialize Flask app
app = Flask(__name__)
model_directory_triage = "./triage_bert"
model_directory = "./model_t5"  # Adjust the path to match where you've saved the model
model_directory_relevant = "./relevantQA_bert"

device = "cuda" if torch.cuda.is_available() else "cpu"

load_model = T5ForConditionalGeneration.from_pretrained(model_directory)
load_model.to(device)
load_tokenizer = T5Tokenizer.from_pretrained(model_directory)
load_model = load_model.eval()

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

def answer_question(question):
    load_model.eval()
    with torch.no_grad():
        input_ids = load_tokenizer(question, return_tensors="pt", padding=True, truncation=True, max_length=512).input_ids.to(device)
        outputs = load_model.generate(
            input_ids,
            max_length=512,
            temperature=0.4,
            do_sample=True,
            top_k=20
        )
        answer = load_tokenizer.decode(outputs[0], skip_special_tokens=True)
        points = answer.split('. ')
        formatted_points = '</p><p>'.join(points)
        # Join the points with <br> tags and wrap in <p> tags
        formatted_answer = '<p>' + '</p><p>'.join(points) + '</p>'
         # Capitalize the first word of the paragraph
        formatted_answer = formatted_answer.capitalize()
        # Remove any occurrence of "Regards"
        formatted_answer = formatted_answer.replace("Regards", "")
        formatted_answer = formatted_answer.replace("thank you", "")
        # Remove the "Click on this link" part
        formatted_answer = formatted_answer.split('Click on this link')[0]
    return formatted_answer

# Route for the home page
@app.route('/')
def home():
    return render_template('pages/index.html')

# Route for handling form submission
@app.route('/qa', methods=['GET', 'POST'])
def qa():
    generated_response = None
    user_message = ""

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

    return render_template('pages/index.html', generated_response=generated_response, user_message=user_message)


# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
