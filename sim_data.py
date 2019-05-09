from flask import Flask, request, Response, make_response, jsonify
from flask_cors import CORS
import numpy as np
import cv2
import base64
app = Flask(__name__)
CORS(app)
timestep = 1
skip_interval=1
@app.route('/upload', methods=['POST']) # for live simulated env
def upload():
    global timestep
    timestep += 1
    resp = Response(response="response_pickled", status=200, mimetype="application/json")
    if timestep%skip_interval != 0:
        return resp

    # print(list(request.form.keys()))
    img = request.form['Image']
    counter = request.form['Counter']
    keypressed = request.form['KeyPressed']
    conditioning_data = None
    if 'x_pos' in request.form.keys():
        xpos = request.form['x_pos']
        ypos = request.form['y_pos']
        zpos = request.form['z_pos']
        xrotation = request.form['x_rot']
        yrotation = request.form['y_rot']
        zrotation = request.form['z_rot']
        conditioning_data = (xpos, ypos, zpos, xrotation, yrotation, zrotation)

    img = base64.b64decode(img)
    nparr = np.frombuffer(img, np.uint8)
    cv2_img = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)

    # result = online_train(cv2_img, conditioning_data)
    # if result==1:
    cv2.imwrite("./training_data/%d.jpg"%timestep, cv2_img)

    return resp

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8002, threaded=False)
