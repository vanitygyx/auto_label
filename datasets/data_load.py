import json
import random

def read_dev_json(input_file,out_test_file): 
    id = 0
    with open(input_file,'r',encoding='utf-8') as f:
        while True:
            d = f.readline()
            if not d:
                break
            line = json.loads(d)
            #print(line)
            data = {}
            data["id"] = id
            data["text"] = line["text"]
            write_data = json.dumps(data,ensure_ascii=False)
            with open(out_test_file,'a',encoding='utf-8') as out:
                out.write(write_data)
                out.write('\n')
            id+=1

def read_predict_json(input_file1,input_file2,output_file): 
    out_text = []
    out_predict_label = []
    out_True_label = []
    with open(input_file1,'r',encoding='utf-8') as f:
        while True:
            d = f.readline()
            if not d:
                break
            line = json.loads(d)
            out_text.append(line["text"])
            out_True_label.append(line["label"])
    with open(input_file2,'r',encoding='utf-8') as f:
        while True:
            entities = []
            predict_lable = {}
            d = f.readline()
            if not d:
                break
            line = json.loads(d)
            id = line["id"]
            entities = line["entities"]
            text = out_text[id]
            for en in entities:
                if en[0] not in predict_lable:
                    predict_lable[en[0]] = {}
                    predict_lable[en[0]][text[en[1]:en[2]+1]]=[[en[1],en[2]]]
                else:
                    if text[en[1]:en[2]+1] in predict_lable[en[0]]:
                        predict_lable[en[0]][text[en[1]:en[2]+1]].append([en[1],en[2]])
                    else:
                        predict_lable[en[0]][text[en[1]:en[2]+1]]=[[en[1],en[2]]]
            out_predict_label.append(predict_lable)
    for i in range(len(out_text)):
        data1 = {}
        data2 = {}
        data3 = {}
        data1["text"] = out_text[i]
        data2["predict_label"] = out_predict_label[i]
        data3["Ture_label"] = out_True_label[i]
        write_data1 = json.dumps(data1,ensure_ascii=False)
        write_data2 = json.dumps(data2,ensure_ascii=False)
        write_data3 = json.dumps(data3,ensure_ascii=False)
        with open(output_file,'a',encoding='utf-8',newline="\n") as out:
                out.write(write_data1)
                out.write('\n')
                out.write(write_data2)
                out.write('\n')
                out.write(write_data3)
                out.write('\n')
                out.write('\n')


if __name__ == "__main__":
    #read_dev_json("datasets/dev.json","datasets/test.json")
    read_predict_json("datasets/dev.json","datasets/test_prediction.json","datasets/out.json")