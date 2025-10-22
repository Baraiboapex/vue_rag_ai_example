const express = require("express");
const cors = require("cors");

require("dotenv").config();

const http = require("http");
const ex_app = express();
const ragAiChat = require("./api/ragAiChat");
const chatServer = http.createServer(ex_app);
const {Server} = require("socket.io");
const io = new Server(chatServer,
    {
        cors:{
            origin:"*",
            methods:["GET","POST"]
        }
    }
);

ex_app.use(cors({
    origin:"*",
    allowedHeaders:"Content-Type",
    methods:"POST"
}));

ragAiChat(io);

chatServer.on("upgrade", (req, socket, head) => {
  console.log("Upgrade requested:", req.url);
});

ex_app.set("port", JSON.parse(process.env.CURRENT_PORT) || 4001);

chatServer.listen(ex_app.get("port"), ()=>{
    console.log("Listening on port " + process.env.CURRENT_PORT);
});