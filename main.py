import cv2
from flask import Flask , jsonify , request , send_file
import PIL.Image
import dlib
import os

app=Flask(__name__)

detector = dlib.cnn_face_detection_model_v1('dogHeadDetector.dat')


@app.route('/check', methods=['POST'])
def disp():
    data=request.files['image']
    img = PIL.Image.open(data.stream)
    img.save(str(data.filename))
    img = cv2.imread(str(data.filename))
    os.remove(str(data.filename))
    dets = detector(img, upsample_num_times=1)
    left = dets[0].rect.left()
    top = dets[0].rect.top()
    right = dets[0].rect.right()
    bottom = dets[0].rect.bottom()
    img = cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0), 2)
    cv2.imwrite('return.jpg',img)
    return send_file('return.jpg', attachment_filename='return.jpg', mimetype='image/jpg')

    #return jsonify({'message':'success'})



@app.route('/', methods=['GET'])
def hello():
    return jsonify({'message':'hello'})

if __name__ == '__main__':
    app.run(debug=True,port=8080)