import ML_model

#val = ML_model.test_call()
#print(val) 

################ Meta Data ######################
#category={0:'Grade 1',1:'Grade 2',2: 'Grade 3', 3: 'Grade 4'}
#category_size={0:'<<Large>>',1:'<<Small>>'}
#################################################
final_result_s1 = ML_model.spots_Analyzer('s1')
print("S1 > " + final_result_s1[0])
final_result_s2 = ML_model.spots_Analyzer('s2')
print("S2 > " + final_result_s2[0])
final_result_s3 = ML_model.spots_Analyzer('s3')
print("S3 > " + final_result_s3[0])
final_result_s4 = ML_model.spots_Analyzer('s4')
print("S4 > " + final_result_s4[0])


final_grade_of_mango = ML_model.final_grade(final_result_s4[1],'s4')

print("Final Grade => "+ final_grade_of_mango)



