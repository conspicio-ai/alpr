from pyrebase import pyrebase
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore, storage
import datetime
import math as m
import time
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
app = firebase_admin.initialize_app(cred, {
    'storageBucket': 'falcon-vision-b63de.appspot.com',
}, name='storage')
bucket = storage.bucket(app=app)
st = firebase.storage()


def push_data(gate, view, AuthID, reg_number, if_reg, time_date, veh_type, visits):
	users_ref = db.collection(AuthID).document('gates').collection(gate).document('view').collection(view).document()
	# docs = users_ref.stream()

	local_path = '/home/himanshu/sih_number_plate/images/send_to_cloud.png'
	path_on_cloud = "vehicle_photos/" +reg_number+"_"+time_date+".png"
	st.child(path_on_cloud).put(local_path)
	blob = bucket.blob(path_on_cloud)
	url = blob.generate_signed_url(datetime.timedelta(hours=123456), method='GET')
	users_ref.set({
		'number': str(reg_number),
		'url':url,
		'registered':if_reg,
		'time':time_date,
		'type':veh_type,
		'visit':visits
		})

def collection_push_data(AuthID,reg_number, gate, view, time_date):
	data_ref = db.collection(AuthID)
	docs = data_ref.stream()
	for i,doc in enumerate(docs):
		doc = doc.to_dict()
		if i == 0:
			place = doc['installationPlace']
			break

	users_ref = db.collection(reg_number).document()
	# docs = users_ref.stream()
	users_ref.set({
		'time':time_date[:5],
		'date':time_date[7:],
		'gate_name':gate,
		'view':view,
		'place':place
		})

def pull_data(AuthID, number):
	data_ref = db.collection(AuthID).document('reg_vehicle').collection('verified')
	docs = data_ref.stream()

	# print(docs)
	for doc in docs:
		# print(doc)
		doc = doc.to_dict()
		# print(doc)
		veh_number = doc['number']
		visits = doc['visits']
		block = doc['block']
		if veh_number == number:
			visit = visits + 1
			users_ref = db.collection(AuthID).document('reg_vehicle').collection('verified').document(veh_number)
			users_ref.set({
				'number': veh_number,
				'visits':visit,
				'block' :block
				})
			return True, visit, block
	return False, 0, False

def get_time():
	t = time.localtime()

	hrs = t.tm_hour
	if hrs<=9:
		hrs = str(hrs)
		hrs = '0' + hrs
	else:
		hrs = str(hrs)

	mins = t.tm_min
	if mins<=9:
		mins = str(mins)
		mins = '0' + mins
	else:
		mins = str(mins)

	day = t.tm_mday
	if day<=9:
		day = str(day)
		day = '0' + day
	else:
		day = str(day)

	month = t.tm_mon
	if month<=9:
		month = str(month)
		month = '0' + month
	else:
		month = str(month)

	year = str(t.tm_year)

	time_local = hrs+':'+mins
	# print(time_local)
	date_local = day + '-' + month + '-' + year
	# print(date_local)
	time_date = time_local +'__'+ date_local
	return time_date
