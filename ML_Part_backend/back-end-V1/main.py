import pymysql
from app import app
from db_config import mysql
from flask import jsonify
from flask import flash, Flask, request
#from flask_restful import Resource, Api
from datetime import date


@app.route('/process')
def getProcessId():
        try:
                
                conn = mysql.connect()
                cursor = conn.cursor(pymysql.cursors.DictCursor)
                cursor.execute("SELECT MAX(process_id) process_id FROM process_data LIMIT 1")
                rows = cursor.fetchall()
                resp = jsonify(rows)
                resp.status_code = 200
                return resp
        except Exception as e:
                return e
        finally:
                cursor.close()
                conn.close()


@app.route('/create-process', methods =['POST'])
def createProcess():
        try:
                today = date.today()
                d1 = today.strftime("%Y-%m-%d")
                sql = "INSERT INTO process_data(date) VALUES(%s)"
                data = (d1,)
                conn = mysql.connect()
                cursor = conn.cursor()
                cursor.execute(sql, data)
                conn.commit()
                resp = jsonify('Process added successfully!')
                resp.status_code = 200
                return resp
        except Exception as e:
                return e
        finally:
                cursor.close()
                conn.close()

@app.route('/home')
def getProcess():
        return jsonify({'text':'Hello World!'})

@app.errorhandler(404)
def not_found(error=None):
    message = {
        'status': 404,
        'message': 'Not Found: ' + request.url,
    }
    resp = jsonify(message)
    resp.status_code = 404

    return resp
	
'''class Mangoes(Resource):
        def get(self):
                return {'mangoes': [{'id':1, 'name':'Balram'},{'id':2, 'name':'Tom'}]} 

api.add_resource(Mangoes, '/mangoes') # Route_1 '''


if __name__ == "__main__":
    app.run()
