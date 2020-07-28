import os
# classes = ['Ambulance','Bicycle','Bus','Car','Limousine','Motorcycle','Taxi','Truck','Van']
classes = ['plate']
for c in classes:
    top_dir = '/home/himanshu/Downloads/train/Vehicle_registration_'+c
    print(top_dir)
    files = os.listdir( top_dir )

    for index,item in enumerate(files):
        if os.path.isdir( os.path.join(top_dir,item) ):
           files.pop(index)

    files.sort()

    duplicates = []
    last_index = None
    for index,item in enumerate(files):

        last_index = index
        extension = ""
        if '.' in item:
            extension = '.' + item.split('.')[-1]
        old_file = os.path.join(top_dir,item)
        new_file = os.path.join(top_dir,str(c+'_'+str(index)) + extension  )
        # while os.path.isfile(new_file):
        #       print("In loop")
        #       last_index += 1
        #       new_file = os.path.join(top_dir,str(classes+'_'+last_index) + extension  )
        print( old_file + ' renamed to ' + new_file ) 
        os.rename(old_file,new_file)