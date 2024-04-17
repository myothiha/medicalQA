# Import necessary modules
from flask import Flask, render_template, request
# from gpt import answer_question
from transformers import AutoModelForCausalLM,AutoTokenizer,pipeline
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
# Initialize Flask app
app = Flask(__name__)

model_directory = "./model_t5"  # Adjust the path to match where you've saved the model
device = "cuda" if torch.cuda.is_available() else "cpu"
load_model = T5ForConditionalGeneration.from_pretrained(model_directory)
load_model.to(device)
load_tokenizer = T5Tokenizer.from_pretrained(model_directory)
load_model = load_model.eval()

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
        formatted_answer = f"<p>{formatted_points}</p><p>Click on this link to ask me a DIRECT QUERY: http://bit.ly/Dr-Sudhir-kumar</p>"
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
        generated_response = answer_question(user_message)

    return render_template('pages/index.html', generated_response=generated_response, user_message=user_message)


# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
