const os = require("os");
const NUMBER_OF_CORES_ON_MACHINE = os.cpus().length - 1;

module.exports = {
  NUMBER_OF_CORES_ON_MACHINE
};