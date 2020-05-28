from tkinter import *
root=Tk()
import cv2
root.configure(background='grey')
entryname=0
root1=0
label123=Label(root1,text="WELCOME TO SECURO")
label1234=(root1)
buton1=Button(root1,text="Reset",font='Times 20',bg='black',fg='Red')
sta=0
l=[]
def initial():
    global root1
    global entryname
    root1=Tk()
    root1.configure(background='grey')
    root1.title("WELCOME TO CREATE PASSWORD MENU")
    label=Label(root1,text="WELCOME TO SECURO")
    label.config(font='Times 45',fg='Red',bg='Orange')
    label.place(relx=0.25,y=0.25)
    entryname=Entry(root1,font='Times 15')
    buton1=Button(root1,text="Done",font='Times 20',bg='black',fg='Red',command=createb)
    buton1.place(x=160,y=240,relx=0.25,rely=0.25,height=30,width=80)
    entryname.place(x=160,y=200,relx=0.25,rely=0.25)
    root1.mainloop()
def createdel():
    global buton1
    global label1234
    entryname.delete(0,'end')
    buton1.destroy()
    buton1=Button(root1,text="Done",font='Times 20',bg='black',fg='Red',command=createb)
    buton1.place(x=160,y=240,relx=0.25,rely=0.25,height=30,width=80)
    label1234=Label(text="",font='Times 35',fg='grey',bg='grey')
    label1234.place(x=90,y=200)
def createb():
    import csv
    global root1
    global entryname
    global buton1
    Id=''
    import cv2
    global label1234
    global l
    with open('users.csv','r') as csv_file:
        csv_reader=csv.reader(csv_file)
        for line in csv_reader:
            for k in line:
                try:
                    l.append(int(k))
                except:
                    print("its user")
        print(l)
    try:
        Id=int(entryname.get())
        print(Id)
        l.append(Id)
        print("l in here ",l)
        with open('users.csv','w') as c:
            fieldnames=['users']
            writ=csv.DictWriter(c,fieldnames=fieldnames)
            writ.writeheader()
            for i in l:
                writ.writerow({'users':i})
        cam = cv2.VideoCapture(0)
        detector=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        sampleNum=0
        while(True):
            ret, img = cam.read()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(gray, 1.3, 5)
            for (x,y,w,h) in faces:
                cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2) 
                sampleNum=sampleNum+1
                cv2.imwrite("dataSet/User."+str(Id)+'.'+ str(sampleNum) + ".jpg", gray[y:y+h,x:x+w])
                cv2.imshow('frame',img) 
            if cv2.waitKey(100) & 0xFF == ord('q'):
                break
            elif sampleNum>500:
                break
        cam.release()
        cv2.destroyAllWindows()
        buton1=Button(root1,text="Reset",font='Times 20',bg='black',fg='Red',command=createdel)
        buton1.place(x=160,y=240,relx=0.25,rely=0.25,height=30,width=80)
        import cv2,os
        import numpy as np
        from PIL import Image
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        detector= cv2.CascadeClassifier("haarcascade_frontalface_default.xml");
        def getImagesAndLabels(path):
            imagePaths=[os.path.join(path,f) for f in os.listdir(path)] 
            faceSamples=[]
            Ids=[]   
            for imagePath in imagePaths:
                if(os.path.split(imagePath)[-1].split(".")[-1]!='jpg'):
                    continue
                pilImage=Image.open(imagePath).convert('L')
        
                imageNp=np.array(pilImage,'uint8')

                Id=int(os.path.split(imagePath)[-1].split(".")[1])
        
                faces=detector.detectMultiScale(imageNp)
    
                for (x,y,w,h) in faces:
                    faceSamples.append(imageNp[y:y+h,x:x+w])
                    Ids.append(Id)
            return faceSamples,Ids
        faces,Ids = getImagesAndLabels('dataSet')
        recognizer.train(faces, np.array(Ids))
        recognizer.save('trainner/trainner.yml')
        print("done")
    except:
        label1234=Label(root1,text="please enter the integer value",font='Times 35',fg='Red',bg='black')
        label1234.place(x=90,y=200)
        print("not integer value")
        buton1=Button(root1,text="Reset",font='Times 20',bg='black',fg='Red',command=createdel)
        buton1.place(x=160,y=240,relx=0.25,rely=0.25,height=30,width=80)

def fianl():
    global l
    global label
    import numpy as np
    import serial
    import cv2
    import csv
    global root
    with open('users.csv','r') as csv_file:
        csv_reader=csv.reader(csv_file)
        for line in csv_reader:
            try:
                if(line[0]!='user'):
                    l.append((line[0]))
            except:
                print("no problem")

    import numpy as np
    import serial
    import cv2
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    cap = cv2.VideoCapture(0)
    rec = cv2.face.LBPHFaceRecognizer_create();
    rec.read("trainner\\trainner.yml")
    id=0
    c=0
    m=0
    stat=0
    font=cv2.FONT_HERSHEY_SIMPLEX
    while 1:
        ret, img = cap.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.5, 5)
        for (x,y,w,h) in faces:
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
            id,conf=rec.predict(gray[y:y+h,x:x+w])
            print(id)
            if conf<65:
                id='found'
                stat=1
                cv2.putText(img,id,(x-w,y-h),font,0.5,(0,225,255),2,cv2.LINE_AA)
                break
            else:
                id='unknown'
                stat=2
                cv2.putText(img,id,(x-w,y-h),font,0.5,(0,225,255),2,cv2.LINE_AA)            
        cv2.imshow('img',img)
        if(stat==1):
            cap.release()
            label1=Label(root,text="LOGIN SUCCESFUL WELCOME!!",font='Times 35',fg='Red',bg='black')
            label1.place(x=20,y=280,relx=0.25,rely=0.25)
            break
        else:

            label2=Label(root,text="LOGIN UNSUCCESFUL",font='Times 35',fg='Red',bg='black')
            label2.place(x=0,y=280,relx=0.25,rely=0.25)
        if cv2.waitKey(1) == ord('q'):
            break
    #cap.release()

cv2.destroyAllWindows()

label=Label(root,text="WELCOME TO SECURO")
label.config(font='Times 45',fg='Red',bg='Orange')
label.place(relx=0.25,y=0.25)
buttonw=Button(text="Create a new user to acces",font='Times 20',fg='Red',bg='black',command=initial)
buttonk=Button(text="Access",font='Times 20',fg='Red',bg='black',command=fianl)
#buttonw=Button(text="",font='Times 20',fg='Red',bg='yellow',command=spark)
buttonw.place(x=160,y=200,relx=0.25,rely=0.25,height=30,width=400)
buttonk.place(x=160,y=240,relx=0.25,rely=0.25,height=30,width=400)
root.geometry('1000x1000+650+350')
root.mainloop()
