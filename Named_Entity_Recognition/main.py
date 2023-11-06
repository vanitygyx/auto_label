from flask import Flask, request, jsonify
from tools.to_test import text_process
from run_ner_crf import main
app = Flask(__name__)

@app.route('/ner', methods=['POST'])
def ner():
    text = request.json.get('text', '')
    split_text= text_process(text)
    entities = main(split_text)
    return jsonify({'entities': entities})
if __name__ == '__main__':      
    app.run(debug=True)
    