from flask import Flask, request, jsonify
from tools.to_test import NER_text_process
from run_ner_crf_new import ner_test
from utils.to_test_fine_grad import RE_text_process
from utils.auto_extraction import re_test

app = Flask(__name__)

@app.route('/ner', methods=['POST'])
def ner():
    text = request.json.get('text', '')
    sub_splite_text,split_text= NER_text_process(text)
    entities = ner_test(sub_splite_text,split_text)
    return jsonify({'entities': entities})

@app.route('/re', methods=['POST'])
def re():
    text = request.json.get('text', '')
    entities = request.json.get('entities', [])
    ent1_list,ent2_list,text_list,ent1_id,ent2_id= RE_text_process(text,entities)
    rel_list = re_test("test.pth",text_list,ent1_list,ent2_list)
    relations =[]
    for i in range(len(rel_list)):
        relations.append({"from_id":ent1_id[i],"to_id":ent2_id[i],"type":rel_list[i]})
    return jsonify({'entities': entities,'relations':relations})
if __name__ == '__main__':      
    app.run(debug=True)