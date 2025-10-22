
const _ = require("lodash");
const {NUMBER_OF_CORES_ON_MACHINE} = require("./environmentHelpers");
const WorkerPool = require("./workerPool");

const createNewWorkerPool = () => {
    const pool = new WorkerPool({
      numThreads: NUMBER_OF_CORES_ON_MACHINE,
    });
    return pool;
};

const ragAiPool = createNewWorkerPool();

function createRagAiChat(io){
    try{

        const ragAiChat = io.of("/ragAiChat");
        
        ragAiChat.on("upgrade", (req, socket, head) => {
            console.log("Upgrade requested:", req.url);
        });

        ragAiChat.on("connection",(socket)=>{
            console.log("TEST");
            socket.on("sendMessage",(message)=>{
                console.log("TEST");
                const query = message.question;
                console.log(query);
                ragAiPool.addNewWorker({workerName:"ragAiTask", workerData:query});
                ragAiPool.runTask(query, (err, result) => {
                    if(err){
                        console.log("Something went wrong : ", err);
                        ragAiPool.close();
                        throw new Error(err);
                    }else{
                        socket.emit("sendResponse", res);
                        ragAiPool.close();
                    }
                });
            });

            socket.on("disconnect", () => {
                console.log("rag chat closed");
            });
        });
        
    }catch(err){
        console.log("Error constructing chat server : ", err);
    }
}

module.exports = createRagAiChat;