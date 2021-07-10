import cv2,os
import numpy as np
import statistics
from keras.layers import Dense #fully connected layers
from keras.models import Sequential

category={0:'Grade 1',1:'Grade 2',2: 'Grade 3', 3: 'Grade 4'}
category_size={0:'<<Large>>',1:'<<Small>>'}
def load_FFNN():
    model=Sequential()
    #an empty Neural Network

    model.add(Dense(16,input_dim=8,activation='relu'))
    #1st Hidden Layer
    model.add(Dense(8,input_dim=16,activation='relu'))
    #2nd Hideen Layer
    model.add(Dense(4,input_dim=8,activation='softmax'))

    model.compile(loss='categorical_crossentropy',optimizer='adam',
                  metrics=['accuracy'])
    

    model.load_weights('Mango_v1.h5')
    return model

def load_FFNN_Size():
    model_size=Sequential()
    #an empty Neural Network

    model_size.add(Dense(8,input_dim=2,activation='relu'))
    #1st Hidden Layer
    model_size.add(Dense(4,input_dim=8,activation='relu'))
    #2nd Hideen Layer
    model_size.add(Dense(2,input_dim=4,activation='softmax'))

    model_size.compile(loss='categorical_crossentropy',optimizer='adam',
                  metrics=['accuracy'])
    

    model_size.load_weights('Mango_Size_v1.h5')
    return model_size

def test_call():
    return 258

def spots_size_Analyzer_python_UI(side_of_mango):
    #test_path='TestImages/'+side_of_mango
    test_path='../front-end/src/assets/images/'+side_of_mango
    img_names=os.listdir(test_path)

    model=load_FFNN()
    model_size = load_FFNN_Size()

    category={0:'Grade 1',1:'Grade 2',2: 'Grade 3', 3: 'Grade 4'}
    category_size={0:'<<Large>>',1:'<<Small>>'}
    ## Final result variable ####
    myResult='' 
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

        #cv2.waitKey()
        #data.append([rm,rsd,gm,gsd,bm,bsd,YCrCb_Ym,YCrCb_Ysd])
        cc = [rm,rsd,gm,gsd,bm,bsd,YCrCb_Ym,YCrCb_Ysd]
        #print(cc)   (3)
        
        result=model.predict([[[rm,rsd,gm,gsd,bm,bsd,YCrCb_Ym,YCrCb_Ysd]]])
        
        label=np.argmax(result,axis=1)[0]
        prob=np.max(result,axis=1)[0]
        prob=round(prob,2)*100
        #print("ddddddddddddddddddddddddddddddd result: ",result, " np.argmax(result,axis=1) :", np.argmax(result,axis=1) )  (4)
        img=cv2.resize(img,(376,251))
        img[200:251,:]=[7,80,227] # bottom bar
        img[:25,:]=[7,80,227]# Top bar
        img[70:100,300:372]=[0,255,0] # grade box
        
        label_size = -1
        if(label == 0 or label == 1): 
            #print("llllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllll")  (5)
            x,y,w,h = cv2.boundingRect(selectedContour)
            cv2.rectangle(img, (x, y), (x + w, y + h), (36,255,12), 4)
            
            #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
            # ########################## Size Analyzer ##################################################
            #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
            
            result_size=model_size.predict([[[h,w]]])
            label_size=np.argmax(result_size,axis=1)[0]
            prob_size=np.max(result_size,axis=1)[0]
            prob_size=round(prob_size,2)*100
            #print("#############>> result_size: ",result_size, " np.argmax(result_size,axis=1) :", np.argmax(result_size,axis=1) )   (5)
            #print(category_size[label_size])  (6)
            
            #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
            #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
            #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
            
            img=cv2.resize(img,(752,502))
            cv2.putText(img,"Height(pixel): "+str(h)+ " Width(pixel): "+str(w),(20,420),cv2.FONT_HERSHEY_SIMPLEX,0.5,
                    (255,255,255),2)
            
        else:
            img=cv2.resize(img,(752,502))
            cv2.putText(img, "Height: No Need Width: No Need",(20,420),cv2.FONT_HERSHEY_SIMPLEX,0.5,
                    (255,255,255),2)

         
        #Spots Analyzer (Classifier 1)
        cv2.putText(img,"Spots Analyzer (Classifier 1)",(55,40),cv2.FONT_HERSHEY_SIMPLEX,0.75,
                    (255,255,255),2)
        
        #Size Analyzer 
        if(label == 0 or label == 1):
            cv2.putText(img,"Spots Analyzer (Classifier 1) --> Size Analyzer ",(55,40),cv2.FONT_HERSHEY_SIMPLEX,0.75,
                    (255,255,255),2)
          
            
        ############ grade box ###################################################
        
        if(label == 0 or label == 1):
            if(label == 0 and label_size == 0):
                cv2.putText(img,str(category[0]),(610,175),cv2.FONT_HERSHEY_SIMPLEX,1,
                    (255,255,255),2)
                myResult = str(category[0]) 
            elif(label == 0 and label_size == 1):
                cv2.putText(img,"Grade 2",(610,175),cv2.FONT_HERSHEY_SIMPLEX,1,
                    (255,255,255),2)
                img[215:275,600:744]=[0,200,0]
                cv2.putText(img,"<<Small>>",(610,250),cv2.FONT_HERSHEY_SIMPLEX,0.70,
                    (255,255,255),2)
                myResult = 'Grade 2 <<Small>>' 
            elif(label == 1 and label_size == 1):
                cv2.putText(img,"Grade 2",(610,175),cv2.FONT_HERSHEY_SIMPLEX,1,
                    (255,255,255),2)
                img[215:275,600:744]=[0,200,0]
                cv2.putText(img,"<<Small>>",(610,250),cv2.FONT_HERSHEY_SIMPLEX,0.70,
                    (255,255,255),2)
                myResult = 'Grade 2 <<Small>>'  
            elif(label == 1 and label_size == 0):
                cv2.putText(img,"Grade 2",(610,175),cv2.FONT_HERSHEY_SIMPLEX,1,
                    (255,255,255),2)
                img[215:275,600:744]=[0,200,0]
                cv2.putText(img,"<<Large>>",(610,250),cv2.FONT_HERSHEY_SIMPLEX,0.70,
                    (255,255,255),2)
                myResult = 'Grade 2 <<Large>>' 
        else:
            cv2.putText(img,str(category[label]),(610,175),cv2.FONT_HERSHEY_SIMPLEX,1,
                    (255,255,255),2)
            myResult = str(category[label]) 
                
        ##########################################################################
        
        cv2.putText(img,"Prediction Probability :"+str(prob),(370,420),cv2.FONT_HERSHEY_SIMPLEX,0.5,
                    (255,255,255),2)
        #--------------------------------------------------------------------------------------------------------
        cv2.putText(img,"R-m         :"+str(rm),(20,440),cv2.FONT_HERSHEY_SIMPLEX,0.35,
                    (255,255,255),1)
        cv2.putText(img,"R-sd        :"+str(rsd),(370,440),cv2.FONT_HERSHEY_SIMPLEX,0.35,
                    (255,255,255),1)
        cv2.putText(img,"G-m         :"+str(gm),(20,455),cv2.FONT_HERSHEY_SIMPLEX,0.35,
                    (255,255,255),1)
        cv2.putText(img,"G-sd        :"+str(gsd),(370,455),cv2.FONT_HERSHEY_SIMPLEX,0.35,
                    (255,255,255),1)
        cv2.putText(img,"B-m         :"+str(bm),(20,467),cv2.FONT_HERSHEY_SIMPLEX,0.35,
                    (255,255,255),1)
        cv2.putText(img,"B-sd        :"+str(bsd),(370,467),cv2.FONT_HERSHEY_SIMPLEX,0.35,
                    (255,255,255),1)
        cv2.putText(img,"YCrCb_Y-m  :"+str(YCrCb_Ym),(20,479),cv2.FONT_HERSHEY_SIMPLEX,0.35,
                    (255,255,255),1)
        cv2.putText(img,"YCrCb_Y-sd :"+str(YCrCb_Ysd),(370,479),cv2.FONT_HERSHEY_SIMPLEX,0.35,
                    (255,255,255),1)
        
        cv2.imshow('TJC Mango Grading System (UCSC)-2020 ',img)
        cv2.waitKey(1000)
        #+str(rsd)+str(gm)+str(gsd)+str(bm)+str(bsd)+str(YCrCb_Ym)+str(YCrCb_Ysd)
        #print(result,label,prob)
    return myResult
        
def spots_Analyzer(side_of_mango):
    
    #test_path='TestImages/'+side_of_mango
    test_path='../front-end/src/assets/images/'+side_of_mango

    img_names=os.listdir(test_path)
    print("calling spots_Analyzer => image name => ")
    print(img_names)
    model=load_FFNN()
    model_size = load_FFNN_Size()

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

        #cv2.waitKey()
        #data.append([rm,rsd,gm,gsd,bm,bsd,YCrCb_Ym,YCrCb_Ysd])
        cc = [rm,rsd,gm,gsd,bm,bsd,YCrCb_Ym,YCrCb_Ysd]
        #print(cc)   (3)
        
        result=model.predict([[[rm,rsd,gm,gsd,bm,bsd,YCrCb_Ym,YCrCb_Ysd]]])
        
        label=np.argmax(result,axis=1)[0]
        prob=np.max(result,axis=1)[0]
        prob=round(prob,2)*100
        
        myResult = str(category[label]) 
        Prediction_Probability = str(prob)         
        ##########################################################################
        
    return [myResult,label,Prediction_Probability]


        
def size_Analyzer(side_of_mango):
    #test_path='TestImages/'+side_of_mango
    test_path='../front-end/src/assets/images/'+side_of_mango
    
    img_names=os.listdir(test_path)

    model=load_FFNN()
    model_size = load_FFNN_Size()
    category_size={0:'<<Large>>',1:'<<Small>>'}
    ## Final result variable ####
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
            
        result_size=model_size.predict([[[h,w]]])
        label_size=np.argmax(result_size,axis=1)[0]
        prob_size=np.max(result_size,axis=1)[0]
        prob_size=round(prob_size,2)*100
        #print("#############>> result_size: ",result_size, " np.argmax(result_size,axis=1) :", np.argmax(result_size,axis=1) )   (5)
        #print(category_size[label_size])  (6)
        myResult=category_size[label_size]
        myProb = str(prob_size)    
        #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
        #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
        #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
    return [myResult,label_size,myProb]   

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
    

