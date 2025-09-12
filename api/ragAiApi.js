const bodyParser = require("body-parser");
const express = require("express");
const { spawn } = require('child_process');

const app = express.Router();

const {NUMBER_OF_CORES_ON_MACHINE} = require("./environmentHelpers");
const WorkerPool = require("./workerPool");

const createNewWorkerPool = () => {
    const pool = new WorkerPool({
      numThreads: NUMBER_OF_CORES_ON_MACHINE,
    });
  
    return pool;
};

const ragAiPool = createNewWorkerPool();

app.post("/sendEmail",async (req, res)=>{
    try{
        const query = req.body;
        ragAiPool.addNewWorker({workerName:"ragAiTask", workerData:query});
        ragAiPool.runTask(emailData, (err, result) => {
            if(err){
                console.log(err);
                throw new Error(err);
            }
            res.status(200).send(result);
            pool.close();
        });
    }catch(err){
        res.status(500).send(JSON.stringify({
            success:false,
            errorMessage:err
        })); 
    }
});

module.exports = app;