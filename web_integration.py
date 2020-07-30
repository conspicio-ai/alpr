from pyrebase import pyrebase
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
# print('loaded')

cred =credentials.Certificate('/home/himanshu/falcon-vision-b63de-firebase-adminsdk-v0rsh-f8c0e2ffe3.json')
default_app = firebase_admin.initialize_app(cred)

db = firestore.client()

config = {
    'apiKey': "AIzaSyCv1rDrMCUUC1AGn0wEdnMpvyzaVm3yrQM",
    'authDomain': "falcon-vision-b63de.firebaseapp.com",
    'databaseURL': "https://falcon-vision-b63de.firebaseio.com",
    'projectId': "falcon-vision-b63de",
    'storageBucket': "falcon-vision-b63de.appspot.com",
    'messagingSenderId': "219989519360",
    'appId': "1:219989519360:web:acfd2b7c7ba4c7d2907ce5",
    'measurementId': "G-L3E99M25MV"
    }
firebase = pyrebase.initialize_app(config)
# db = firebase.database()

def push_data(Gate, view, AuthID, reg_number, if_reg, time, veh_type, visits):
	users_ref = db.collection(AuthID).document('gates').collection(Gate).document('view').collection(view).document()
	# docs = users_ref.stream()
	users_ref.set({
		'number': str(reg_number),
		'registered':if_reg,
		'time':time,
		'type':veh_type,
		'visit':visits
		})

def pull_data(AuthID, number):
	data_ref = db.collection(AuthID).document('reg_vehicle').collection('verified')
	docs = data_ref.stream()


	for doc in docs:
		print(doc)
		doc = doc.to_dict()
		print(doc)
		veh_number = doc['number']
		visits = doc['visits']
		if veh_number == number:
			visits += 1
			return True, visits
	return False, 0


# for doc in docs:
#     print(f'{doc.id} => {doc.to_dict()}')
