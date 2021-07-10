import ML_model
import threading
################ Meta Data ######################
#category={0:'Grade 1',1:'Grade 2',2: 'Grade 3', 3: 'Grade 4'}
#category_size={0:'<<Large>>',1:'<<Small>>'}
#################################################

final_result_s1 = []
final_result_s2 = []
final_result_s3 = []
final_result_s4 = []

 
def S1_Result(): 
    final_result_s1.append(ML_model.spots_Analyzer('s1'))
    print("S1 > " + final_result_s1[0][0])
  
def S2_Result(): 
    final_result_s2.append(ML_model.spots_Analyzer('s2'))
    print("S2 > " + final_result_s2[0][0])
    
def S3_Result(): 
    final_result_s3.append(ML_model.spots_Analyzer('s3'))
    print("S3 > " + final_result_s3[0][0])
    
def S4_Result(): 
    final_result_s4.append(ML_model.spots_Analyzer('s4'))
    print("S4 > " + final_result_s4[0][0])


  
 
# creating thread 
t1 = threading.Thread(target=S1_Result) 
t2 = threading.Thread(target=S2_Result)
t3 = threading.Thread(target=S3_Result)
t4 = threading.Thread(target=S4_Result)
  
# starting thread 1 
t1.start() 
# starting thread 2 
t2.start()
# starting thread 3 
t3.start()
# starting thread 4 
t4.start() 
  
# wait until thread 1 is completely executed 
t1.join() 
# wait until thread 2 is completely executed 
t2.join()
# wait until thread 3 is completely executed 
t3.join()
# wait until thread 4 is completely executed 
t4.join() 

if (final_result_s1[0][1] == 3 or
    final_result_s2[0][1] == 3 or
    final_result_s3[0][1] == 3 or
    final_result_s4[0][1] == 3 ):
    print("Final Grade => "+ "Grade 4")
elif(final_result_s1[0][1] == 2 or
    final_result_s2[0][1] == 2 or
    final_result_s3[0][1] == 2 or
    final_result_s4[0][1] == 2 ):
    print("Final Grade => "+ "Grade 3")
elif(final_result_s1[0][1] == 1 or
    final_result_s2[0][1] == 1 or
    final_result_s3[0][1] == 1 or
    final_result_s4[0][1] == 1 ):
    final = ML_model.final_grade(1,'s2')  ## Apply Size Analyzer
    print("Final Grade => "+ final)
else:
    final = ML_model.final_grade(0,'s2')  ## Apply Size Analyzer
    print("Final Grade => "+ final)
    
# All the threads completely executed 
print("Done!")


# instruction for versions
'''
I am trying to use the tensorflow/keras as server with multithreading.
Same error with:
keras==2.3.1
tensorflow==2.0.0
Solve the issue with:
tensorflow==1.15 @caigen
https://github.com/keras-team/keras/issues/13353
'''




