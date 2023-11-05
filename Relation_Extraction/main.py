from flask import Flask, request, jsonify
from utils.to_test_fine_grad import text_process
from utils.auto_extraction import test
app = Flask(__name__)

@app.route('/ner', methods=['POST'])
def ner():
    text = request.json.get('text', '')
    entities = request.json.get('entities', [])
    ent1_list,ent2_list,text_list,ent1_id,ent2_id= text_process(text,entities)
    rel_list = test("Model_Parameter/test.pth",text_list,ent1_list,ent2_list)
    relations =[]
    for i in range(len(rel_list)):
        relations.append({"from_id":ent1_id[i],"to_id":ent2_id[i],"type":rel_list[i]})
    return jsonify({'entities': entities,'relations':relations})
if __name__ == '__main__':      
    app.run(debug=True)