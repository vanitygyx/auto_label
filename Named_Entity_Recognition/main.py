from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/re', methods=['POST'])
def re():
    text = request.json.get('text', '')
    ent1_list,ent2_list,text_list,ent1_id,ent2_id= text_process(text,entities)
    rel_list = test("test.pth",text_list,ent1_list,ent2_list)
    relations =[]
    for i in range(len(rel_list)):
        relations.append({"from_id":ent1_id[i],"to_id":ent2_id[i],"type":rel_list[i]})
    return jsonify({'entities': entities})
if __name__ == '__main__':      
    app.run(debug=True)
    