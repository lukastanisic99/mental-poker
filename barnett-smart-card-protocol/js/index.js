const wasm = require('../pkg/barnett_smart_card_protocol.js');

// console.log("Exported functions and fields:");
// console.log(Object.keys(wasm.__wasm));

async function run() {
  console.log("Running in Nodejs START");
  console.log(wasm.protocl());
  console.log("Running in Nodejs END");
}

run();

