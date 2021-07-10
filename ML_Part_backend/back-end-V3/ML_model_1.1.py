import cv2,os
import numpy as np
import statistics

category={0:'Grade 1',1:'Grade 2',2: 'Grade 3', 3: 'Grade 4'}
category_size={0:'<<Large>>',1:'<<Small>>'}

#using KNN        
def spots_Analyzer(side_of_mango):
    
    #test_path='TestImages/'+side_of_mango
    test_path='../front-end/src/assets/images/'+side_of_mango

    img_names=os.listdir(test_path)
    print("calling spots_Analyzer => image name => ")
    print(img_names)
   
    category={0:'Grade 1',1:'Grade 2',2: 'Grade 3', 3: 'Grade 4'}
    
    ## Final result variable ####
    myResult=''
    Prediction_Probability = ''
    label = 404
    
    for img_name in img_names:
        img_path=os.path.join(test_path,img_name)
        img=cv2.imread(img_path)
        image=cv2.resize(img,(376,251))
        ore = image.copy()
        ore2 = image.copy()
        ore3 = image.copy()
        ore5 = image.copy()
        ore6 = image.copy()
        ore7 = image.copy()
        ######### No need ############
        #ore4 = image.copy()
        ##############################

        rows,cols,channels = image.shape

        original = image.copy()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower = np.array([22, 93, 0], dtype="uint8")
        upper = np.array([45, 255, 255], dtype="uint8")
        orgmask = cv2.inRange(image, lower, upper)



        mask = cv2.GaussianBlur(orgmask,(5,5),0)


        cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        cv2.drawContours(ore2, cnts, -1, (0,255,0),3)

        ############ No need ###############
        #cntsTest = cv2.findContours(orgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        #cntsTest = cntsTest[0] if len(cntsTest) == 2 else cntsTest[1]
        #cv2.drawContours(ore4, cntsTest, -1, (0,255,0),3)
        ####################################

        for c in cnts:
            if(cv2.contourArea(c) > 700):
                selectedContour = c
                x,y,w,h = cv2.boundingRect(c)
                cv2.rectangle(ore3, (x, y), (x + w, y + h), (36,255,12), 2)

        for i in range(rows):
            for j in range(cols):
                if cv2.pointPolygonTest(selectedContour,(j, i),True)<0:
                     original[i, j]= (0, 0, 0)
                    #dist = cv2.pointPolygonTest(cnt,(50,50),True)
                else:
                    original[i, j]= (255, 255, 255)


        #+++++++++++ openning ++++++++++++++++++++++++++++++++++++
        originalOpen = original.copy()
        kernel = np.ones((17,17),np.uint8)
        originalOpenning = cv2.morphologyEx(originalOpen, cv2.MORPH_OPEN, kernel)
        
        #cv2.imshow('opening', originalOpenning)
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        #print(mask)                    (1)
        #print(originalOpenning)        (2)

        originalOpenning = cv2.cvtColor(originalOpenning, cv2.COLOR_BGR2GRAY)

        originalOpenning = cv2.GaussianBlur(originalOpenning,(3,3),0)

        cnts2 = cv2.findContours(originalOpenning, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts2 = cnts2[0] if len(cnts2) == 2 else cnts2[1]
        cv2.drawContours(ore5, cnts2, -1, (0,255,0),3)
        
        selectedContour
        for c in cnts2:
            if(cv2.contourArea(c) > 700):
                selectedContour = c
                x,y,w,h = cv2.boundingRect(c)
                cv2.rectangle(ore6, (x, y), (x + w, y + h), (36,255,12), 2)

        #$$$$$$$$$$$$$$$$$$$$ hist $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
        rVal =[]
        bVal =[]
        gVal =[]
        YCrCb_YVal = []

        ######CrCb -> b ######
        BGR2YCrCb = ore7.copy()
        YCrCb = cv2.cvtColor(BGR2YCrCb, cv2.COLOR_BGR2YCrCb)
        Y, Cr, Cb  = cv2.split(YCrCb)
        YCrCb_Y = Y
        ######################

        b, g, r = cv2.split(ore7)
        #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
        for i in range(rows):
            for j in range(cols):
                if cv2.pointPolygonTest(selectedContour,(j, i),True)<0:
                    ore7[i, j]= (255, 255, 255)
                    #dist = cv2.pointPolygonTest(cnt,(50,50),True)
                else:
                    #$$$$$$$$$$$$$$$$$$ hist $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
                    rVal.append(int(r[i][j]))
                    bVal.append(int(b[i][j]))
                    gVal.append(int(g[i][j]))
                    YCrCb_YVal.append(int(YCrCb_Y[i][j]))
                    #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
                    
        
        #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        rm = statistics.mean(rVal)
        rsd = statistics.stdev(rVal)
        #print("(R)Mean  : ",rm)
        #print("(R)Standard Deviation of Sample set is " ,rsd)

        gm = statistics.mean(gVal)
        gsd = statistics.stdev(gVal)
        #print("(G)Mean  : ",gm)
        #print("(G)Standard Deviation of Sample set is " ,gsd)

        bm = statistics.mean(bVal)
        bsd = statistics.stdev(bVal)
        #print("(B)Mean  : ",bm)
        #print("(B)Standard Deviation of Sample set is " ,bsd)

        ##########CrCb -> b#############  to reduce shining issue
        YCrCb_Ym = statistics.mean(YCrCb_YVal)
        YCrCb_Ysd = statistics.stdev(YCrCb_YVal)
        #print("(YCrCb_Y)Mean  : ",YCrCb_Ym)
        #print("(YCrCb_Y)Standard Deviation of Sample set is " ,YCrCb_Ysd)
        ################################

        #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        
        cc = [rm,rsd,gm,gsd,bm,bsd,YCrCb_Ym,YCrCb_Ysd]
        #print(cc)   
        
        #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
        import joblib
        # loading we trained algo
        algo = joblib.load('KNN_model.sav')
        #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
       
        result = algo.predict([[rm,rsd,gm,gsd,bm,bsd,YCrCb_Ym,YCrCb_Ysd]])
        label= result
        myResult = str(category[label[0]])
        
        ##########################################################################
    
    return [myResult,label]


        
def size_Analyzer(side_of_mango):
    #test_path='TestImages/'+side_of_mango
    test_path='../front-end/src/assets/images/'+side_of_mango
    
    img_names=os.listdir(test_path)
    category_size={0:'<<Large>>',1:'<<Small>>'}
    myResult=''
    myProb = ''
    label_size = 404
    for img_name in img_names:
        img_path=os.path.join(test_path,img_name)
        img=cv2.imread(img_path)
        image=cv2.resize(img,(376,251))
        ore = image.copy()
        ore2 = image.copy()
        ore3 = image.copy()
        ore5 = image.copy()
        ore6 = image.copy()
        ore7 = image.copy()
        ######### No need ############
        #ore4 = image.copy()
        ##############################

        rows,cols,channels = image.shape

        original = image.copy()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower = np.array([22, 93, 0], dtype="uint8")
        upper = np.array([45, 255, 255], dtype="uint8")
        orgmask = cv2.inRange(image, lower, upper)



        mask = cv2.GaussianBlur(orgmask,(5,5),0)


        cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        cv2.drawContours(ore2, cnts, -1, (0,255,0),3)

        ############ No need ###############
        #cntsTest = cv2.findContours(orgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        #cntsTest = cntsTest[0] if len(cntsTest) == 2 else cntsTest[1]
        #cv2.drawContours(ore4, cntsTest, -1, (0,255,0),3)
        ####################################

        for c in cnts:
            if(cv2.contourArea(c) > 700):
                selectedContour = c
                x,y,w,h = cv2.boundingRect(c)
                cv2.rectangle(ore3, (x, y), (x + w, y + h), (36,255,12), 2)

        for i in range(rows):
            for j in range(cols):
                if cv2.pointPolygonTest(selectedContour,(j, i),True)<0:
                     original[i, j]= (0, 0, 0)
                    #dist = cv2.pointPolygonTest(cnt,(50,50),True)
                else:
                    original[i, j]= (255, 255, 255)


        #+++++++++++ openning ++++++++++++++++++++++++++++++++++++
        originalOpen = original.copy()
        kernel = np.ones((17,17),np.uint8)
        originalOpenning = cv2.morphologyEx(originalOpen, cv2.MORPH_OPEN, kernel)
        
        #cv2.imshow('opening', originalOpenning)
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        #print(mask)                    (1)
        #print(originalOpenning)        (2)

        originalOpenning = cv2.cvtColor(originalOpenning, cv2.COLOR_BGR2GRAY)

        originalOpenning = cv2.GaussianBlur(originalOpenning,(3,3),0)

        cnts2 = cv2.findContours(originalOpenning, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts2 = cnts2[0] if len(cnts2) == 2 else cnts2[1]
        cv2.drawContours(ore5, cnts2, -1, (0,255,0),3)
        
        selectedContour
        for c in cnts2:
            if(cv2.contourArea(c) > 700):
                selectedContour = c
                x,y,w,h = cv2.boundingRect(c)
                

        #---------------------------
        #print("llllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllll")  (5)
        x,y,w,h = cv2.boundingRect(selectedContour)
        
        #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
        # ########################## Size Analyzer ##################################################
        #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
        import joblib
        # loading we trained algo
        algo = joblib.load('KNN_model_size.sav')
        #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
       
        result = algo.predict([[h,w]])
        label= result
        myResult = str(category[label[0]])

        
        #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
        #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
        #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
    return [myResult,label]   

def final_grade(final_result_of_spots_analyzer,side_of_mango):
    myResult = ''
    label = final_result_of_spots_analyzer
    if(label == 0 or label == 1):
        label_size = size_Analyzer(side_of_mango)[1]
        if(label == 0 and label_size == 0):
            myResult = str(category[0]) 
        elif(label == 0 and label_size == 1):
            myResult = 'Grade 2 <<Small>>' 
        elif(label == 1 and label_size == 1):
            myResult = 'Grade 2 <<Small>>'  
        elif(label == 1 and label_size == 0):
            myResult = 'Grade 2 <<Large>>' 
    else:
        myResult = str(category[label]) 
    return myResult            




#print(spots_Analyzer("s1")[0])  

