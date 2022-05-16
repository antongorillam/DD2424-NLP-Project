from flask import Flask, jsonify, request
from flask_restful import Api, Resource
from flask_cors import CORS
import load

app = Flask(__name__)
api = Api(app)

class Synthesize(Resource):
    def get(self):
        num_words = request.args.get("num_words")
        initial_in = request.args.get("initial_word")
        model = load.loadModel()
        response = jsonify(message=model.synthesize(initial_input=initial_in, seq_length=int(num_words)))
        response.headers.add("Access-Control-Allow-Origin", "*")
        return response

@app.route('/')
def parse_request():
    return "Hello World"

api.add_resource(Synthesize, "/Synthesize")

if __name__ == '__main__':
    app.run(port="8081")
