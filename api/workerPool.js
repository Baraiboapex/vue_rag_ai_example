const { AsyncResource } = require('node:async_hooks');
const { EventEmitter } = require('node:events');
const path = require('node:path');
const { Worker } = require('node:worker_threads');
const _ = require("lodash");

const kTaskInfo = Symbol('kTaskInfo');
const kWorkerFreedEvent = Symbol('kWorkerFreedEvent');

class WorkerPoolTaskInfo extends AsyncResource {
  constructor(callback) {
    super('WorkerPoolTaskInfo');
    this.callback = callback;
  }

  done(err, result) {
    this.runInAsyncScope(this.callback, null, err, result);
    this.emitDestroy();  // `TaskInfo`s are used only once.
  }
}

class WorkerPool extends EventEmitter {
  constructor({
    numThreads,
    workers
}) {
    super();
    this.numThreads = numThreads;
    //this.workers = {};
    this.workers = [];
    this.freeWorkers = [];
    this.tasks = [];

    const hasWorkers = workers !== undefined && !(_.isEmpty(workers)) && workers !== null;
    const hasNumThreads = numThreads !== undefined && numThreads !== null && numThreads !== 0;
    const amountOfWorkersWithinAcceptableThredRange = hasWorkers && (hasNumThreads ? workers.length <= numThreads : false);

    if(amountOfWorkersWithinAcceptableThredRange){
        for (let i = 0; i < workers.length; i++){
            const data = {
                workerName:workers[i].workerName,
                workerData:workers[i].workerData
            };
            this.addNewWorker(data);
        } 
    }
        
    // Any time the kWorkerFreedEvent is emitted, dispatch
    // the next task pending in the queue, if any.
    this.on(kWorkerFreedEvent, () => {
      if (this.tasks.length > 0) {
        const { task, callback } = this.tasks.shift();
        this.runTask(task, callback);
      }
    });
  }
  addNewWorker({
    workerName,
    workerData
}){
    try{
      const worker = new Worker(path.resolve(__dirname, workerName+'.js'), {
          workerData:(workerData !== undefined ? workerData : undefined)
      });
      worker.on('message', (result) => {
        const taskInfo = worker[kTaskInfo];

        if (!taskInfo) {
          this.emit("No task info!");
          return;
        }

        if (message && message.type === "end") {
          taskInfo.done(null, null); 

          worker[kTaskInfo] = null;
          this.freeWorkers.push(worker);
          this.emit(kWorkerFreedEvent);
        }else{
          this.emit("done");
        }
      });
      worker.on('error', (err) => {
        // In case of an uncaught exception: Call the callback that was passed to
        // `runTask` with the error.
        if (worker[kTaskInfo]){
          worker[kTaskInfo].done(err, null);
        }else{
          console.log("ERROR IN POOL", err);
            this.emit('error', err);
        }
        // Remove the worker from the list and start a new Worker to replace the
        // current one.
        this.workers.splice(this.workers.indexOf(worker), 1);
        this.addNewWorker({
            workerName,
            workerData
        });
      });
      this.workers.push(worker);
      this.freeWorkers.push(worker);
      this.emit(kWorkerFreedEvent);
    }catch(err){
      console.log("POOL ERROR!!!! ==> ",err);
    }
  }

  runTask(task, callback) {
    if (this.freeWorkers.length === 0) {
      // No free threads, wait until a worker thread becomes free.
      this.tasks.push({ task, callback });
      return;
    }

    const worker = this.freeWorkers.pop();
    worker[kTaskInfo] = new WorkerPoolTaskInfo(callback);
    worker.postMessage(task);
  }

  close() {
    for (const worker of this.workers){
      worker.terminate();
    } 
  }
}

module.exports = WorkerPool;