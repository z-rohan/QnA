from flask import Flask, render_template, request, jsonify
from answerMe import AnsweringModel


app = Flask(__name__)

@app.route('/', methods=['GET'])
def redner_temp():
    # if request.method == 'POST':
    #     obj = AnsweringModel()
    #     context = request.form.get('para')
    #     question = request.form.get('que')
    #     ans = obj.answer_question(question, context)
    #     return render_template('answer.html', answer=ans)
    if request.method == 'GET':
        return render_template('index.html') 

@app.route('/get_answer', methods=['POST'])
def get_answer():
    if request.method == 'POST':
        obj = AnsweringModel()
        context = request.form.get('para')
        question = request.form.get('que')
        ans, conf = obj.answer_question(question, context)
        
        if conf > 0.4:
            return jsonify({'answer': ans})
        else:
            return jsonify({'answer': 'Out of context or not able to find the answer in the context'})

if __name__== '__main__':
    app.run(debug=True)