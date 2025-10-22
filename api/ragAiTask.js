const {spawn} = require("child_process");
const {
    isMainThread, 
    workerData, 
    parentPort
  } = require('node:worker_threads');

  if (!isMainThread) {
    const query = workerData.query;
    const path = require('path');
    const scriptPath = path.resolve(__dirname, '../rag_ai.py');

    const runRagAI = spawn('python', [scriptPath, query]);

    let finalOutput = "";

    runRagAI.stdout.on('data', (data) => {
        finalOutput = data;
        runRagAI.kill();
    });

    runRagAI.stderr.on('error', (data) => {
        throw new Error(`Error: ${data}`);
    });

    runRagAI.on("close",(code)=>{
        if(code === 0){
            console.log(finalOutput);
            data = JSON.parse(finalOutput);
            parentPort.postMessage(data);
        }
    });
  }