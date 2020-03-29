import cv2
import matplotlib.pyplot as plt
import os
import numpy as np
from werkzeug.utils import secure_filename
from flask import Flask, request
from flask_cors import CORS
import inferProgram
import uuid
import json

app = Flask(__name__)
CORS(app)
@app.route('/inferFourFumula', methods=['POST'])
def infer():
    f = request.files['img1']
    # 保存图片
    save_father_path = 'images'
    img_path = os.path.join(save_father_path, str(uuid.uuid1()) + '.' + secure_filename(f.filename).split('.')[-1])
    if not os.path.exists(save_father_path):
        os.makedirs(save_father_path)
    f.save(img_path)
    r = inferProgram.FourOperation(5).infer_image(img_path)[1]
    os.remove(img_path)
    r = json.dumps(r)
    r = '{"result": '+str(r)+'}'
    return r

@app.route('/inferMatrix', methods=['POST'])
def inferMartix():
    f = request.files['img1']
    # 保存图片
    save_father_path = 'images'
    img_path = os.path.join(save_father_path, str(uuid.uuid1()) + '.' + secure_filename(f.filename).split('.')[-1])
    if not os.path.exists(save_father_path):
        os.makedirs(save_father_path)
    f.save(img_path)
    try:
        r = inferProgram.Martix(37).infer_image(img_path)[1]
        r = inferProgram.ToolBar.getMartrix().getResult(r)
    except Exception:
        r = 'ERROR'
    os.remove(img_path)
    r = json.dumps(r)
    r = '{"result": '+str(r)+'}'
    return r

@app.route('/inferUCnumber', methods=['POST'])
def inferUnk():
    f = request.files['img1']
    # 保存图片
    save_father_path = 'images'
    img_path = os.path.join(save_father_path, str(uuid.uuid1()) + '.' + secure_filename(f.filename).split('.')[-1])
    if not os.path.exists(save_father_path):
        os.makedirs(save_father_path)
    f.save(img_path)
    r = inferProgram.Unk(66).infer_image(img_path)[1]
    print(r)
    os.remove(img_path)
    r = json.dumps(r)
    r = '{"result": '+str(r)+'}'
    return r

@app.route('/inferAcc', methods=['POST'])
def inferAcc():
    f = request.files['img1']
    # 保存图片
    save_father_path = 'images'
    img_path = os.path.join(save_father_path, str(uuid.uuid1()) + '.' + secure_filename(f.filename).split('.')[-1])
    if not os.path.exists(save_father_path):
        os.makedirs(save_father_path)
    f.save(img_path)
    r = inferProgram.Acc(101).infer_image(img_path)[1]
    print(r)
    os.remove(img_path)
    r = json.dumps(r)
    r = '{"result": '+str(r)+'}'
    return r

@app.route('/inferIntegral', methods=['POST'])
def inferIntegral():
    f = request.files['img1']
    # 保存图片
    save_father_path = 'images'
    img_path = os.path.join(save_father_path, str(uuid.uuid1()) + '.' + secure_filename(f.filename).split('.')[-1])
    if not os.path.exists(save_father_path):
        os.makedirs(save_father_path)
    f.save(img_path)
    try:
        r = inferProgram.Integral(151).infer_image(img_path)[1]
        r = inferProgram.ToolBar.getIntegral().getResult(r)
    except Exception:
        r = 'ERROR'
    print(r)
    os.remove(img_path)
    r = json.dumps(r)
    r = '{"result": '+str(r)+'}'
    return r

@app.route('/inferDerivative', methods=['POST'])
def inferDerivative():
    f = request.files['img1']
    # 保存图片
    save_father_path = 'images'
    img_path = os.path.join(save_father_path, str(uuid.uuid1()) + '.' + secure_filename(f.filename).split('.')[-1])
    if not os.path.exists(save_father_path):
        os.makedirs(save_father_path)
    f.save(img_path)
    try:
        r = inferProgram.Unk(66).infer_image(img_path)[1][0]
        r = inferProgram.ToolBar.getDerivative().getResult(r)
    except Exception:
        r = 'ERROR'
    print(r)
    os.remove(img_path)
    r = json.dumps(r)
    r = '{"result": '+str(r)+'}'
    return r

ip1 = '127.0.0.1'
ip2 = '172.16.7.238'
if __name__ == '__main__':
    # 启动服务，并指定端口号
    app.run(host = str(ip1),port=870)