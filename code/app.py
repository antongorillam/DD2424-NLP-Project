from flask import Flask, jsonify
from flask_restful import Api, Resource
import load

app = Flask(__name__)
api = Api(app)
var cors = require('cors')

class Synthesize(Resource):
    def get(self):
        model = load.loadModel()
        response = jsonify(message=model.synthesize(initial_input="l", seq_length=100))
        response.headers.add("Access-Control-Allow-Origin", "*")
        return response

@app.route('/')
def hello_world():
    return "wassup"

api.add_resource(Synthesize, "/Synthesize")

if __name__ == '__main__':
    app.run(cors())
