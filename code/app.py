from flask import Flask, jsonify
from flask_restful import Api, Resource
import load

app = Flask(__name__)
api = Api(app)
var cors = require('cors')

class Synthesize(Resource):
    def get(self):
        print("Loading")
        model = load.loadModel()
        print("synthesize")
        response = jsonify(message=model.synthesize(initial_input="l", seq_length=100))
        response.headers.add("Access-Control-Allow-Origin", "*")
        print("went here")
        return response

api.add_resource(Synthesize, "/Synthesize")

if __name__ == '__main__':
    app.run(cors())
