from flask import Flask, render_template, request
import numpy as np
import re
import base64
import cv2
from mindspore import load_checkpoint, load_param_into_net
from mindspore import Tensor, Model
from mindspore import dtype as mstype
from mindspore.nn.metrics import Accuracy
from LeNet5 import *

app = Flask(__name__)

model_file = './lenet5.ckpt'
param_dict = load_checkpoint(model_file)
# 加载参数到网络中
net = LeNet5()
load_param_into_net(net, param_dict)
net_loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
global model
model = Model(net, net_loss, metrics={'accuracy': Accuracy()}) 

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/predict/', methods=['Get', 'POST'])
def preditc():
    parseImage(request.get_data())
    img = cv2.imdecode(np.fromfile('output.png', dtype=np.uint8), 0)
    img= np.array(img)
    # img= 255-img[...]
    mean, std = 33.3285, 78.5655 
    img =(img-mean)/std
    img=cv2.resize(img,(28,28))
    img=img.reshape((1,1,28,28))
    output = model.predict(Tensor(img,dtype=mstype.float32))
    predict = np.argmax(output.asnumpy(),axis=1)[0]
    return str(predict)

def parseImage(imgData):
    imgStr = re.search(b'base64,(.*)', imgData).group(1)
    with open('./output.png', 'wb') as output:
        output.write(base64.decodebytes(imgStr))


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=3335)
