const {
    isMainThread, 
    workerData, 
    parentPort
  } = require('node:worker_threads');

  if (!isMainThread) {
    const query = workerData.query;
    const runRagAI = spawn('python', ['rag_ai.py', query]);

    let finalOutput = "";

    runRagAI.stdout.on('data', (data) => {
        finalOutput = data;
    });

    runRagAI.stderr.on('data', (data) => {
        throw new Error(`Error: ${data}`);
    });

    runRagAI.on("close",(code)=>{
        if(code === 0){
            data = JSON.parse(finalOutput);
            parentPort.postMessage(data);
        }
    });
  }