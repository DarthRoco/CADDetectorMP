from flask import Flask, redirect, url_for, request, render_template
#from model import model_predict
import os
from werkzeug.utils import secure_filename
import requests
import cv2,numpy as np
import base64
import tensorflow.keras.models as tfm
app = Flask(__name__, static_url_path='/C:/Users/Shreyas Bhat/Desktop/basicfront/')


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('Frontpage.html')

@app.route('/home', methods=['GET'])
def home():
    # Main page
    return render_template('Frontpage.html')



@app.route('/predict/', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Get the file from post request
        #f = request.files['file']
        f1 = request.files['file']
        # f2 = request.form.get('scale')
        # f3 = request.form.get('image_type')
        
        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f1.filename))
        f1.save(file_path)
        imggg=cv2.imread(file_path)
        img=cv2.resize(imggg,(512,512))
        _,_,rr=cv2.split(img)
        rr[rr>180]=255
        rr[rr<180]=0
        img=rr
        img=cv2.resize(img,(64,64))
        img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
       
        # img=cv2.cvt
        filename=file_path+"OUT.jpg"
        cv2.imwrite(filename,img)
        img=img/255
        img=np.reshape(img,(1,64,64,3))
        # cv2.imshow("Hell",imggg)
        # cv2.waitKey(0)


        # scale = f2
        
        # with open(url_for('static', filename='scale.txt'),"w") as f_scale:
        #     f_scale.write(scale)
        #     f_scale.write(f3)
            
        # img_type = 1 if f3=="camera" else 0
        
        # if img_type==1:
        #     yml_file_path = os.path.join(basepath, 'options/df2k/test_df2k.yml')
        # else:
        #     yml_file_path = os.path.join(basepath, 'options/dped/test_dped.yml')       
        model=tfm.load_model('./CADModel.h5',compile=False)
        k=float(model.predict(img)[0])

        print("\n\n\n\n\n\n\n",k,"\n\n\n\n\n\n")
        # r = requests.post("http://127.0.0.1:8080/predictions/super_res",files={'data':open(file_path,'rb')})
        # print(r,"HEioowhd---------------------------")
        # imgdata = base64.b64decode(r.content)
        # with open(output_path, 'wb') as f:
        # 	f.write(imgdata)
        #r.content.save(output_path)
        if k>0.5:
            return render_template('positive.html',fillename=secure_filename(f1.filename)+"OUT.jpg")
        else:
            return render_template('Negative.html',fillename=secure_filename(f1.filename)+"OUT.jpg")

         #https://stackoverflow.com/questions/46785507/python-flask-display-image-on-a-html-page
    return 'OK'


if __name__ == '__main__':
    app.run(debug=True)