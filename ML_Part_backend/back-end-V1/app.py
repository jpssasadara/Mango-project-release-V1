from flask import Flask
from flask_cors import CORS, cross_origin
#from flask_restful import Resource, Api

app = Flask(__name__)
#api = Api(app)
CORS(app)
