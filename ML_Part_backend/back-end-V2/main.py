import pymysql
from app import app
from db_config import mysql
from flask import jsonify
from flask import flash, Flask, request
#from flask_restful import Resource, Api
from datetime import date
import ML_model_Testing_multi_threding_Env

@app.route('/process')
def getProcessId():
        try:
                
                conn = mysql.connect()
                cursor = conn.cursor(pymysql.cursors.DictCursor)
                cursor.execute("SELECT MAX(process_id) process_id FROM process_data LIMIT 1")
                rows = cursor.fetchone()
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

######### Requesting Result from ML_Model#####################
@app.route('/result')
def getResult():
        try:
                resp = ML_model_Testing_multi_threding_Env.get_Grade()
                return resp
        except Exception as e:
                return e
@app.route('/resultserial')
def getResultSerial():
        try:
                resp = ML_model_Testing_multi_threding_Env.get_Grade_serial_execution()
                return resp
        except Exception as e:
                return e
##############################################################

################ For chart -> to call for WS #################
# Query ->        
'''
SELECT grade, COUNT(*) as count ,process_id 
FROM mango_data 
where process_id= (select max(process_id) from mango_data)
GROUP BY grade
'''

@app.route('/GetMangoCount')
def GetMangoCount():
        try:
                
                conn = mysql.connect()
                cursor = conn.cursor(pymysql.cursors.DictCursor)
                cursor.execute("SELECT grade, COUNT(*) as count,process_id "+
                               "FROM mango_data "+
                               "where process_id= (select max(process_id) from mango_data)" +
                               "GROUP BY grade")
                rows = cursor.fetchall()
                resp = jsonify(rows)
                resp.status_code = 200
                return resp
        except Exception as e:
                return e
        finally:
                cursor.close()
                conn.close()
# Get all Query ->
'''
SELECT grade, COUNT(*) as count 
FROM mango_data 
GROUP BY grade
'''
@app.route('/GetAllMangoCount')
def GetAllMangoCount():
        try:
                
                conn = mysql.connect()
                cursor = conn.cursor(pymysql.cursors.DictCursor)
                cursor.execute("SELECT grade, COUNT(*) as count "+
                               "FROM mango_data "+
                               "GROUP BY grade")
                rows = cursor.fetchall()
                resp = jsonify(rows)
                resp.status_code = 200
                return resp
        except Exception as e:
                return e
        finally:
                cursor.close()
                conn.close()

##############################################################


@app.route('/create_mango/<int:id>', methods=['POST'])
def create_mango(id):
	try:
		conn = mysql.connect() 
		cursor = conn.cursor()
		cursor.execute("INSERT INTO mango_data(grade, process_id) VALUES('grade2 small', %s)", (id,))
		conn.commit()
		resp = jsonify({'message':'Mango created successfully!','side1':'../assets/images/s1/1.jpg',
                                'side2':'../assets/images/s2/2.jpg','side3':'../assets/images/s3/3.jpg',
                                'side4':'../assets/images/s4/4.jpg','top':'../assets/images/s5/5.jpg',
                                'bottom':'../assets/images/s6/6.jpg'})
		resp.status_code = 200
		return resp
	except Exception as e:
		print(e)
	finally:
		cursor.close() 
		conn.close()
		

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
