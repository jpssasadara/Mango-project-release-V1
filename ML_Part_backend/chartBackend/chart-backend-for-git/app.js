const express = require('express');
const socket = require('socket.io');
const bodyParser = require('body-parser');
const cors = require('cors');



const app = express();
app.use(cors({origin: '*'}));
app.use(bodyParser);
let x = true;

// for call python service ==> 
var XMLHttpRequest = require("xmlhttprequest").XMLHttpRequest;
const xhttp = new XMLHttpRequest();

//for python service ==>
countResponse =[];
grade1 = 0;
grade2s = 0;
grade2l = 0;
grade3=0;
grade4 =0;

AllcountResponse =[];
Allgrade1=0;
Allgrade2s =0;
Allgrade2l= 0;
Allgrade3= 0;
Allgrade4= 0;

const server = app.listen(3000,() => {
    console.log('Started in 3000');
});

const io = socket(server);

io.sockets.on('connection', (socket) => {
    console.log(`new connection id: ${socket.id}`);
    sendData(socket);
})

function sendData(socket){
    
    if(x){
        //socket.emit('data1', Array.from({length: 5}, () => Math.floor(Math.random() * 590)+ 10));
        GetmangoCount();
        socket.emit('data1', countResponse);
        x = !x;
    }else{
        //socket.emit('data2', Array.from({length: 5}, () => Math.floor(Math.random() * 590)+ 10));
        GetAllmangoCount();
        socket.emit('data2', AllcountResponse);
        x = !x;
    }
    console.log(`data is ${x}`);
    setTimeout(() => {
        sendData(socket);
    }, 1000);
}

// ################  Calling for python service to get mango count ###############################
function GetmangoCount(){

    this.grade1=0;
    this.grade2s =0;
    this.grade2l= 0;
    this.grade3= 0;
    this.grade4= 0;


    xhttp.open("GET", `http://localhost:5000/GetMangoCount`, false);
    xhttp.send();
    const data = JSON.parse(xhttp.responseText);
    /* [ { count: 1, grade: 'grade2 small', process_id: 104 },
         { count: 1, grade: 'grade1', process_id: 104 }   ]         */
    console.log(data);

    data.forEach(element => {
        console.log(element['grade']);
        if(element['grade'] == "grade1"){
            this.grade1 = element['count'];
        }else if (element['grade']=="grade2 small") {
            this.grade2s = element['count'];
        } else if (element['grade']=="grade2 large") {
            this.grade2l = element['count'];
        } else if (element['grade']=="grade3") {
            this.grade3 = element['count'];
        } else {
            console.log("elas part");
            this.grade4 = element['count'];
        }
    });
    // [ 1, 1, 0, 0, 0 ]
    this.countResponse =[this.grade1, this.grade2s, this.grade2l, this.grade3, this.grade4];
    console.log(this.countResponse);
}

// ################  Calling for python service to get mango count ###############################
function GetAllmangoCount(){

    this.Allgrade1=0;
    this.Allgrade2s =0;
    this.Allgrade2l= 0;
    this.Allgrade3= 0;
    this.Allgrade4= 0;

    xhttp.open("GET", `http://localhost:5000/GetAllMangoCount`, false);
    xhttp.send();
    const data = JSON.parse(xhttp.responseText);
    /* [ { count: 2, grade: 'grade1' },
         { count: 73, grade: 'grade2 small' } ]       */
    console.log(data);

    data.forEach(element => {
        console.log(element['grade']);
        if(element['grade'] == "grade1"){
            this.Allgrade1 = element['count'];
        }else if (element['grade']=="grade2 small") {
            this.Allgrade2s = element['count'];
        } else if (element['grade']=="grade2 large") {
            this.Allgrade2l = element['count'];
        } else if (element['grade']=="grade3") {
            this.Allgrade3 = element['count'];
        } else {
            console.log("elas part");
            this.Allgrade4 = element['count'];
        }
    });
    // [ 1, 1, 0, 0, 0 ]
    this.AllcountResponse =[this.Allgrade1, this.Allgrade2s, this.Allgrade2l, this.Allgrade3, this.Allgrade4];
    console.log(this.AllcountResponse);
}
// Angular part => ['Grade 1', 'Grade 2 S', 'Grade 2 L', 'Grade 3', 'Grade 4']