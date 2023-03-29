const video = document.querySelector('video');

// request access to webcam
navigator.mediaDevices.getUserMedia({ video: { width: 426, height: 240 } }).then((stream) => video.srcObject = stream);

// returns a frame encoded in base64

username = window.localStorage.getItem("username")

const WS_URL = 'ws://localhost:8000/recognize';
const FPS = 3;
var socket;

var progress = document.getElementById("progress")
var al = document.getElementById("alert")


var emit;
var canvas = document.getElementById("canvasOutput");
var ctx = canvas.getContext("2d");

function initSocket() {
    var cont = 0;
    socket = new WebSocket(WS_URL);
    socket.onopen = () => {
        console.log(`Connected to ${WS_URL}`);

    }
    socket.onclose = () => {
        console.log("close");
        socket = null

    }
    socket.onmessage = function (event) {
        var data = JSON.parse(event.data);
    
        var image = new Image();
        image.onload = function () {
            ctx.drawImage(image, 0, 0, 300, 150);
        };
        image.height = "420px"
        image.src = "data:image/png;base64,"+data.image
        if (data.validate && cont < 100 ){
            cont = cont + 1
            progress.innerHTML = `${cont}%`
            progress.style.width = `${cont}%`

        }
        if(cont === 100){
            console.log(cont);
            stopButton.click()
            al.style.display = "block"
            al.innerHTML = "Usuario verificado"
        }
    }
}

var startButton = document.getElementById("startButton")
var stopButton = document.getElementById("stopButton")

var type = 'image/png'
startButton.onclick = () => {
    if (socket === null) {
        initSocket()
    }
    emit = setInterval(() => {
        const canvas = document.createElement('canvas');
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        canvas.getContext('2d').drawImage(video, 0, 0);
        var image = canvas.toDataURL(type);
        var data =JSON.stringify({"username":username,"image":image})
        socket.send(data);
    }, 100);
}

stopButton.onclick = () => {
    socket.close()
    clearInterval(emit)
    emit = null;
    ctx.clearRect(0, 0, 300, 150)
}

initSocket()