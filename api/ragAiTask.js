const {spawn} = require("child_process");
const {
    isMainThread, 
    workerData, 
    parentPort
  } = require('node:worker_threads');

  if (!isMainThread) {
    try{
        console.log("STARTING PYTHON BACKGROUND PROCESS...");

        const query = workerData.query;
        console.log("QUERY : ", query);
        const path = require('path');
        const scriptPath = path.resolve(__dirname, '../rag_ai.py');

        console.log("FOUND PYTHON SCRIPT : ", scriptPath);

        const runRagAI = spawn('python', [scriptPath/*, query*/]);

        runRagAI.stdout.on('data', (data) => {
            const rawData = data.toString();
            const chunks = rawData.split('\n').filter(Boolean); 

            for (const chunk of chunks) {
                if (chunk === "<<END_OF_STREAM>>") {
                    parentPort.postMessage({ type: 'end' });
                    return;
                }

                parentPort.postMessage({ type: 'token', content: chunk });
            }
        });

        runRagAI.stderr.on('error', (data) => {
            console.log(`Error in RAG AI : ${data}`);
            // runRagAI.kill();
            throw new Error(`Error: ${data}`);
        });

        runRagAI.on("close",(code)=>{
            console.log("CODE : ", code);
            console.log("PROCESS CLOSED");
        });

    } catch(err){
        console.log("PROCESS ERROR : ",err);
    }
  }