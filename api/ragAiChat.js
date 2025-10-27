
const _ = require("lodash");
const {spawn} = require("child_process");

let runRagAI = null;
let ragAiChat = null;

const currentQueries = new Map();

function attatchResponseListeners(socketId){
    try{
        if (!runRagAI) return;
        runRagAI.stdout.on('data', (data) => {
            const lines = data.toString().split('\n');

            const socketId = Array.from(currentQueries.keys())[0];
            const socket = currentQueries.get(socketId);
            const eofStreamTag = "<<END_OF_STREAM>>";
            
            if (socket) {
                for (const line of lines) {
                    if(!line.includes(eofStreamTag)){
                        console.log(line);
                        socket.emit("sendResponse", line);
                    }
                }
            }
        });

        runRagAI.stderr.on('error', (data) => {
            const currentError = data.toString().trim();
            console.log(`Error in RAG AI : ${currentError}`);

            const currentSocketId = Array.from(currentQueries.keys())[0];
            console.log(`CURRENT SOCKET ID => ${currentSocketId}`);
            const socketId = currentSocketId
            const socket = currentQueries.get(socketId);

            if(socket){
                currentQueries.delete(socketId);
                throw new Error(`Error: ${data}`);
            }
        });
    }catch(err){
        throw new Error("Error while attatching response listeners => Rag AI process was never mounted!")
    }
}

function createRagAiChat(io){
    try{
        if(!runRagAI){
            ragAiChat = io.of("/ragAiChat");
            const path = require('path');
            const scriptPath = path.resolve(__dirname, '../rag_ai.py');

            runRagAI = spawn("python", ["-u", scriptPath]);
            attatchResponseListeners(runRagAI); 
        }
        if(ragAiChat){
            ragAiChat.on("upgrade", (req, socket, head) => {
                console.log("Upgrade requested:", req.url);
            });

            ragAiChat.on("connection",(socket)=>{
                socket.on("sendMessage",(message)=>{
                    if(!runRagAI){
                        throw new Error(`Error: ${data}`);
                    }
                    const query = message.question;

                    currentQueries.set(socket.id, socket);

                    runRagAI.stdin.write(query + "\n");

                    console.log("MESSAGE LOADED");
                });
            });
            
            ragAiChat.on("close_rag_ai",()=>{
                activeQueries.delete(socket.id);
            });
        }else{
            throw new Error("Chat not ready yet.");
        }
    }catch(err){
        console.log("Error constructing chat server : ", err);
    }
}

module.exports = createRagAiChat;