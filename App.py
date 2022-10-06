import os,io,base64
from pickle import TRUE
from flask import Response,session, send_file
from flask_session import Session
from flask import Flask,render_template,request,redirect, url_for
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import false
from werkzeug.utils import secure_filename
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import re
import cv2

UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}


app=Flask(__name__)
app.secret_key='sanjaiswar'
app.config['SQLALCHEMY_DATABASE_URI']='mysql://root:root@localhost/flaskdb'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS']=false
db=SQLAlchemy(app)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config["SESSION_PERMANENT"] = False
app.config["SESSION_TYPE"] = "filesystem"
Session(app)

class Employee(db.Model):
    id=db.Column(db.Integer,primary_key=True)
    name=db.Column(db.String(100))
    email=db.Column(db.String(100))
    phone=db.Column(db.String(100))
    pic=db.Column(db.String(100))

    def __init__(self,name,email,phone,pic):
        self.name =name
        self.email=email
        self.phone=phone
        self.pic=pic

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def Index():
    emp=Employee.query.all()
    return render_template('index.html',employees=emp)

@app.route('/all-emp')
def allEmp():
    emp=Employee.query.all()
    return render_template('employees.html',employees=emp)

@app.route('/new-emp')
def newEmp():
    return render_template('new_emp.html')

@app.route('/create',methods=['GET', 'POST'])
def create():
    if request.method=='POST':
        if 'pic' not in request.files:
            return redirect(url_for('Index'))
        file = request.files['pic']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            return redirect(url_for('Index'))
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        name=request.form['name']
        email=request.form['email']
        phone=request.form['phone']
        pic=filename
        
        emp=Employee(name,email,phone,pic)
        db.session.add(emp)
        db.session.commit()
        return redirect(url_for('Index'))

@app.route('/edit-emp/<id>/',methods=['GET'])
def editEmp(id):
    employee=Employee.query.get(id)
    return render_template('edit_emp.html',emp=employee)

@app.route('/update-emp',methods=['POST'])
def updateEmp():
    if request.method=='POST':
        id=request.form['id']
        emp=Employee.query.get(id)
        if 'pic' in request.files:
            file = request.files['pic']
            # if user does not select file, browser also
            # submit an empty part without filename
            if file.filename != '':
                if file and allowed_file(file.filename):
                    filename = secure_filename(file.filename)
                    file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
                    emp.pic=filename
        emp.name=request.form['name']
        emp.email=request.form['email']
        emp.phone=request.form['phone']
        db.session.commit()
        return redirect(url_for('Index'))

@app.route('/delete-emp/<id>/',methods=['GET','POST'])
def deleteEmp(id):
    employee=Employee.query.get(id)
    db.session.delete(employee)
    db.session.commit()
    return redirect(url_for('Index'))

@app.route('/plot-graph')
def plotGraph():
    return render_template('plot_graph.html')

@app.route('/visualize1')
def visualize1():
    xlist = np.linspace(-3.0, 3.0, 100)
    ylist = np.linspace(-3.0, 3.0, 100)
    X, Y = np.meshgrid(xlist, ylist)
    Z = np.sqrt(X**2 + Y**2)
    fig,ax=plt.subplots(1,1)
    cp = ax.contourf(X, Y, Z)
    fig.colorbar(cp) # Add a colorbar to a plot
    ax.set_title('Filled Contours Plot')
    #ax.set_xlabel('x (cm)')
    ax.set_ylabel('y (cm)')
    canvas=FigureCanvas(fig)
    img = io.BytesIO()
    fig.savefig(img)
    img.seek(0)
    return send_file(img,mimetype='img/png')

@app.route('/visualize2')
def visualize2():
    def f(x, y):
        return np.sin(np.sqrt(x ** 2 + y ** 2))
	
    x = np.linspace(-6, 6, 30)
    y = np.linspace(-6, 6, 30)

    X, Y = np.meshgrid(x, y)
    Z = f(X, Y)

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.contour3D(X, Y, Z, 50, cmap='binary')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_title('3D contour')
    canvas=FigureCanvas(fig)
    img = io.BytesIO()
    fig.savefig(img)
    img.seek(0)
    return send_file(img,mimetype='img/png')

@app.route('/visualize3')
def visualize3():
    fig,ax=plt.subplots(figsize=(5,5))
    ax = fig.add_axes([0,0,1,1])
    langs = ['C', 'C++', 'Java', 'Python', 'PHP']
    students = [23,17,35,29,12]
    ax.bar(langs,students)
    canvas=FigureCanvas(fig)
    img = io.BytesIO()
    fig.savefig(img)
    img.seek(0)
    return send_file(img,mimetype='img/png')

@app.route('/visualize4')
def visualize4():
    fig,ax=plt.subplots(figsize=(5,5))
    ax = fig.add_axes([0,0,1,1])
    ax.axis('equal')
    langs = ['C', 'C++', 'Java', 'Python', 'PHP']
    students = [23,17,35,29,12]
    ax.pie(students, labels = langs,autopct='%1.2f%%')
    canvas=FigureCanvas(fig)
    img = io.BytesIO()
    fig.savefig(img)
    img.seek(0)
    return send_file(img,mimetype='img/png')

@app.route('/visualize5')
def visualize5():
    girls_grades = [89, 90, 70, 89, 100, 80, 90, 100, 80, 34]
    boys_grades = [30, 29, 49, 48, 100, 48, 38, 45, 20, 30]
    grades_range = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    fig = plt.figure()
    ax=fig.add_axes([0,0,1,1])
    ax.scatter(grades_range, girls_grades, color='r')
    ax.scatter(grades_range, boys_grades, color='b')
    ax.set_xlabel('Grades Range')
    ax.set_ylabel('Grades Scored')
    ax.set_title('scatter plot')
    canvas=FigureCanvas(fig)
    img = io.BytesIO()
    fig.savefig(img)
    img.seek(0)
    return send_file(img,mimetype='img/png')

@app.route('/visualize6')
def visualize6():
    plt.rcParams["figure.figsize"] = [5, 5]
    plt.rcParams["figure.autolayout"] = True
    fig,ax=plt.subplots(figsize=(5,5))
    axis = fig.add_subplot(1, 1, 1)
    xs = np.random.rand(100)
    ys = np.random.rand(100)
    axis.plot(xs, ys)
    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)
    return Response(output.getvalue(), mimetype='image/png')

@app.route('/data-eval')
def dataEval():
    return render_template('data_eval.html')

@app.route('/evaluate',methods=['GET', 'POST'])
def evaluate():
    if request.method=='POST':
        gen_words='''an at as are aren't amazing amazed amazing but be because been being can can't could couldn't do don't does does't did didn't else ever every even each either evenly extreme examine examiner example eager energy enough eventually expression exemption entertaining for go goes got give gone given get gets gives hi hello how has his her him hasn't have haven't here is isn't if in it its it's just kiss kissed kissing kind kindly keen keep kept let leave live me must might may more much many mension mention means meaning meanigful no nor niether never not none new next near net name nill newer of on over onto obvious obviously ofcourse pass put perfect pure purely poor poorly please perfectly perfection politely proper properly pay payed payment pending plural policy politics police passing passes passed pour pot plot quite quit question query queen quarrel qwality quantity right refer read rest remind remember real rare raugh root reverse reward return reform request requesting responsible rock rocking river so some same shame soon saw see seen say said sad sorrow shake shook shameful shameless shamelessly soft send sent switch sware summer season second secondary simultaneous simultaneously situation success story sequence signal sound sounds sounding surrounding surround soak solid surprise the to that this then those they them take taken took takes talk talks told tell tells try tries tried up use ur untill unless unseen untold under valid validity vital virus viral vast verbal verdict view views viewing viewer will was whenever when would wouldn't won't wasn't with within wish wishes wishing waste wasted wasting washed wash washing warm warming warn warning warned weather whether willing willingly wise wiser while who whom whoever whose x'mas xerox yes yet yet you your yield yummy yum year young younger yearly zeal'''

        #act_ans='''Inheritance in Java is a concept that acquires the properties from one class to other classes; for example, the relationship between father and son. Inheritance in Java is a process of acquiring all the behaviours of a parent object.'''
        act_ans=request.form['act_ans']
        cand_ans=request.form['cand_ans']
        elevel=request.form['level']
        fromwc=int(request.form['fwc'])
        towc=int(request.form['twc'])

        stats_list1=getMatch(act_ans, cand_ans, elevel, fromwc, towc)
        stats_list2=getMeaningfulWords(3,act_ans,cand_ans,gen_words)
        stats_list=stats_list1 + stats_list2
        #stats_list=stats_list1+stats_list2
        return render_template('eval_result.html',cand_ans=cand_ans, stats_list= stats_list,act_ans=act_ans)

def getMatch(aa, ca, level, fwc, twc):
    stats_list=[]
    if level == 'word': 
        specialChars = ",;." 
        for specialChar in specialChars:
            aa = aa.replace(specialChar, ' ')
            ca = ca.replace(specialChar,' ')
        aa=aa.strip()
        ca=ca.strip()
        ans_arr = ca.split()
        total_words = len(ans_arr)
        for r in range(twc, fwc-1, -1):
            match_count = 0
            for i in range(0, total_words):
                s = ' '.join(ca.split()[i:(r+i)])
                listToStr = ''.join(map(str, s))
                if listToStr.lower().strip() in aa.lower().strip():
                    match_count = match_count+1
                    print(listToStr)
            stats_list.append(f"Total {r} match words score: {match_count}")
        return stats_list
    if level == 'dot':
        specialChars = ",;" 
        for specialChar in specialChars:
            aa = aa.replace(specialChar, ' ')
            ca = ca.replace(specialChar,' ')
        aa=aa.strip()
        ca=ca.strip()
        regex_pattern = r"[.]"
        ans_arr = re.split(regex_pattern, ca)
        match_count = 0
        for paragraphs in ans_arr:
            if paragraphs.lower().strip() in aa.lower().strip():
                match_count = match_count+1
                # print(paragraphs.strip())
        stats_list.append(f"Total matching paragraphs score: {match_count}")
        return stats_list

    if level == 'comma':
        regex_pattern = r"[;,.]"
        ans_arr = re.split(regex_pattern, ca)
        match_count = 0
        for sentence in ans_arr:
            if sentence.lower().strip() in aa.lower().strip():
                match_count = match_count+1
                print(sentence.strip())
        stats_list.append(f"Total matching sentences score: {match_count}")
        return  stats_list

## For matching answers at higher level

def getMeaningfulWords(ws,actans,candans,gen_words):
    stats_list=[]
    specialChars = ",;.()[]<>\{\}" 
    for specialChar in specialChars:
        actans = actans.replace(specialChar, ' ')
        candans = candans.replace(specialChar,' ')
        
    actans=actans.lower().strip()
    candans=candans.lower().strip()
    text_arr = actans.split()
    total_words = len(text_arr)
    match_count = 0
    for i in range(0,total_words):
        s = ' '.join(actans.split()[i:(total_words)])
        s_arr=s.split()
        mwc=0
        wrds_list=[]
        for wrds in s_arr:
            
            if mwc>ws:
                break
            else:
                if wrds in gen_words:
                    wrds_list.append(wrds)
                else:
                    wrds_list.append(wrds)
                    mwc+=1
        mw=' '.join(wrds_list)
        if mw.lower().strip() in candans.lower().strip():
                match_count = match_count+1
                # print(paragraphs.strip())
        
        print(mw.strip())
    stats_list.append(f"Total exact match lines score: {match_count}")
    return stats_list

@app.route('/reg-face')
def regFace():
    return render_template('reg_face.html')

@app.route('/capture-preview',methods=['GET', 'POST'])
def capturePreview():
    if request.method=='POST':
        name = request.form['name'] 
    return render_template('capture_preview.html',name=name)

@app.route('/frame_capture/<name>')
def video_feed(name):
    return Response(createDataset(name), mimetype='multipart/x-mixed-replace; boundary=frame')

def createDataset(name):
    haar_file = 'haarcascade_frontalface_default.xml'
    sub_data=name
    # All the faces data will be
    #  present this folder
    datasets = 'datasets' 
    
    # These are sub data sets of folder,
    # for my faces I've used my name you can
    # change the label here 
    
    path = os.path.join(datasets, sub_data)
    if not os.path.isdir(path):
        os.mkdir(path)
    
    # defining the size of images
    (width, height) = (130, 100)   
    
    #'0' is used for my webcam,
    # if you've any other camera
    #  attached use '1' like this
    face_cascade = cv2.CascadeClassifier(haar_file)
    webcam = cv2.VideoCapture(0)
    
    # The program loops until it has 30 images of the face.
    count = 1
    while count<41:
        (_, im) = webcam.read()
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 4)
        for (x, y, w, h) in faces:
            cv2.rectangle(im, (x, y), (x + w, y + h), (255, 0, 0), 2)
            face = gray[y:y + h, x:x + w]
            face_resize = cv2.resize(face, (width, height))
            cv2.imwrite('% s/% s.png' % (path, count), face_resize)
        count += 1

        # cv2.imshow('OpenCV', im)
        # key = cv2.waitKey(10)
        # if key == 27:
        #     break

        ret, buffer = cv2.imencode('.jpg', im)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
           
@app.route('/detect-face')
def detectFace():
    return render_template('detect_face.html')

@app.route('/detect-preview')
def detect_preview():
    return Response(detectDataset(), mimetype='multipart/x-mixed-replace; boundary=frame')

def detectDataset():
    size = 4
    haar_file = 'haarcascade_frontalface_default.xml'
    datasets = 'datasets'
    
    # Part 1: Create fisherRecognizer
    print('Recognizing Face Please Be in sufficient Lights...')
    
    # Create a list of images and a list of corresponding names
    (images, labels, names, id) = ([], [], {}, 0)
    for (subdirs, dirs, files) in os.walk(datasets):
        for subdir in dirs:
            names[id] = subdir
            subjectpath = os.path.join(datasets, subdir)
            for filename in os.listdir(subjectpath):
                path = subjectpath + '/' + filename
                label = id
                images.append(cv2.imread(path, 0))
                labels.append(int(label))
            id += 1
    (width, height) = (130, 100)
    
    # Create a Numpy array from the two lists above
    (images, labels) = [np.array(lis) for lis in [images, labels]]
    
    # OpenCV trains a model from the images
    # NOTE FOR OpenCV2: remove '.face'
    model = cv2.face.LBPHFaceRecognizer_create()
    model.train(images, labels)
    
    # Part 2: Use fisherRecognizer on camera stream
    face_cascade = cv2.CascadeClassifier(haar_file)
    webcam = cv2.VideoCapture(0)
    while True:
        (_, im) = webcam.read()
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            cv2.rectangle(im, (x, y), (x + w, y + h), (255, 0, 0), 2)
            face = gray[y:y + h, x:x + w]
            face_resize = cv2.resize(face, (width, height))
            # Try to recognize the face
            prediction = model.predict(face_resize)
            cv2.rectangle(im, (x, y), (x + w, y + h), (0, 255, 0), 3)
    
            if prediction[1]<500:
    
                cv2.putText(im, '% s - %.0f' %
        (names[prediction[0]], prediction[1]), (x-10, y-10),
        cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0))
            else:
                cv2.putText(im, 'not recognized',
        (x-10, y-10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0))
    
        # cv2.imshow('OpenCV', im)
        
        # key = cv2.waitKey(10)
        # if key == 27:
        #     break   
        ret, buffer = cv2.imencode('.jpg', im)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


if __name__ =="__main__":
    app.run(debug=True)