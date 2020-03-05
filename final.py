from pymongo import MongoClient
from flask import Flask, render_template
import json
from datetime import date
import requests
import json

client = MongoClient("mongodb://test:test@cluster0-shard-00-00-razp3.mongodb.net:27017,cluster0-shard-00-01-razp3.mongodb.net:27017,cluster0-shard-00-02-razp3.mongodb.net:27017/test?ssl=true&replicaSet=Cluster0-shard-0&authSource=admin&retryWrites=true&w=majority") 
app = Flask(__name__)

@app.route('/')
def real():
    db = client.get_database('number_plate_detection')
    in_record = db.in_
    final_record = db.final
    _in_ = list(in_record.find())
    _final_ = list(final_record.find())
    return render_template('real.html', in_ = _in_, out_ = _final_)


def sendPostRequest(reqUrl, apiKey, secretKey, useType, phoneNo, senderId, textMessage):
  req_params = {
  'apikey':apiKey,
  'secret':secretKey,
  'usetype':useType,
  'phone': phoneNo,
  'message':textMessage,
  'senderid':senderId
  }
  return requests.post(reqUrl, req_params)


def entryGate(NUMBER,ENTRY_DATE,ENTRY_TIME,PHONE_NO,MESSAGE):
    db = client.get_database('number_plate_detection')
    in_record = db.in_
    user_record = db.USER
    _user_ = list(user_record.find())
    flag = 0
    for i in _user_:
        if i['NUMBER'] == NUMBER:
            js = {
                'NUMBER' : NUMBER,
                'ENTRY DATE' : ENTRY_DATE,
                'ENTRY TIME' : ENTRY_TIME
            }
            flag  = 1
            in_record.append(js)
            # send entry message
            sendPostRequest('https://www.sms4india.com/api/v1/sendCampaign', 'TLAC32922MTPG25XVKLZO8SJ46NRVS50', 'N04NP865HY2ERGED', 'stage', i['PHONE_NO'], 'Rishesh Agarwal', MESSAGE )
    if flag != 1:
        # send welcome message
        sendPostRequest('https://www.sms4india.com/api/v1/sendCampaign', 'TLAC32922MTPG25XVKLZO8SJ46NRVS50', 'N04NP865HY2ERGED', 'stage', PHONE_NO, 'Rishesh Agarwal', MESSAGE)
        js = {
                'NUMBER' : NUMBER,
                'ENTRY DATE' : ENTRY_DATE,
                'ENTRY TIME' : ENTRY_TIME
            }
        flag  = 1
        in_record.append(js)
    return
        

def exitGate(NUMBER,EXIT_DATE,EXIT_TIME,GATE_NO,MESSAGE):
    db = client.get_database('number_plate_detection')
    in_record = db.in_
    final_record = db.final
    user_record = db.USER    
    _in_ = list(in_record.find())
    _user_ = list(user_record.find())
    for i in _in_:
        for u in _user_:
            if u['NUMBER'] == NUMBER:
                # send Message
                sendPostRequest('https://www.sms4india.com/api/v1/sendCampaign', 'TLAC32922MTPG25XVKLZO8SJ46NRVS50', 'N04NP865HY2ERGED', 'stage', u['PHONE_NO'], 'Rishesh Agarwal', MESSAGE )

        for i in _user_:
            if i['NUMBER'] == NUMBER:
                ENTRY_DATE = i['DATE']
                ENTRY_TIME = i['TIME']
                
                in_record.delete_one({
                    'NUMBER' : NUMBER,
                    'DATE' : i['DATE'],
                    'TIME' : i['TIME']
                })

                js = {
                    'NUMBER' : NUMBER,
                    'ENTRY DATE' : ENTRY_DATE,
                    'ENTRY TIME' : ENTRY_TIME,
                    'EXIT DATE' : EXIT_DATE,
                    'EXIT TIME' : ENTRY_TIME
                }

                final_record.append(js)
                break
    return

if __name__ == "__main__":
    app.run()