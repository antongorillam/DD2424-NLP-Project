from flask import Flask
from flask_restful import Api, Resource
import load

app = Flask(__name__)
api = Api(app)


class HelloWorld(Resource):
    def get(self):
        return {"data": "Hello World"}

class Synthesize(Resource):
    def get(self):
        print("Loading")
        model = load.loadModel()
        print("synthesize")
        to_ret = model.synthesize(initial_input="l", seq_length=100)
        return {"data": to_ret}

api.add_resource(HelloWorld, "/helloworld")
api.add_resource(Synthesize, "/Synthesize")

if __name__ == '__main__':
    app.run(port=8080, debug=False)
