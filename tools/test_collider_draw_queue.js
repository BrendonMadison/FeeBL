"use strict";
const fs = require("node:fs"),
    path = require("node:path");
// Reuse the bounded DOM/canvas fixtures; the delayed read below drives the real
// HTML renderFrame implementation, including the field note and particle canvas.
const suite = fs.readFileSync(path.join(__dirname, "test_collider_contract.js"), "utf8");
const regression = String.raw`
(async()=>{
 let releaseRead;const heldRead=new Promise(resolve=>{releaseRead=resolve}),reads=[];
 const cleanLong=JSON.parse(fs.readFileSync(path.join(__dirname,'testdata/collider_longitudinal.json'),'utf8'));
 sandbox.queueLong=cleanLong;
 sandbox.queueManifest={...sandbox.fixtureManifest,frames:cleanLong.manifest.frames.map((f,i)=>({...sandbox.fixtureManifest.frames[i],...f}))};
 sandbox.queueRead=async index=>{reads.push(index);if(index===0)await heldRead;return [[[125,0,0,0,0,0]],[[125,0,0,0,0,0]]]};
 vm.runInContext('man=queueManifest;nfr=man.frames.length;e0=man.energy_GeV;scientific=null;diagnostics=null;nativeFields=ColliderFields.validate(queueLong.fields,queueLong.manifest);initializeFieldControls();frame=queueRead;drawBusy=false;drawPending=false;playing=false',sandbox);
 nodes.get('ax').value='z';nodes.get('ay').value='y';nodes.get('view').value='full';nodes.get('fieldMode').value='on';nodes.get('fr').value='0';
 const draining=vm.runInContext('draw()',sandbox);
 assert.deepEqual(reads,[0]);
 assert.equal(vm.runInContext('drawBusy',sandbox),true);
 for(const index of [1,0,1]){nodes.get('fr').value=String(index);vm.runInContext('draw()',sandbox)}
 vm.runInContext('playing=false',sandbox); // no playback timer remains to rescue a dropped redraw
 assert.deepEqual(reads,[0]);
 assert.equal(vm.runInContext('drawPending',sandbox),true);
 releaseRead();await draining;
 assert.deepEqual(reads,[0,1],'busy draws must coalesce to the final requested frame');
 assert.equal(nodes.get('fr').value,'1');
 assert(nodes.get('frlab').textContent.startsWith('1 / 1'));
 assert(nodes.get('fieldNote').textContent.includes('Frame 1: 3/3 source slices sampled (3 active)'));
 assert.equal(nodes.get('fieldLegend').hidden,false);
 assert.equal(nodes.get('ax').value,'z');assert.equal(nodes.get('ay').value,'y');
 assert.equal(vm.runInContext('drawBusy',sandbox),false);
 assert.equal(vm.runInContext('drawPending',sandbox),false);
 // A later nonbusy request still starts an ordinary render.
 await vm.runInContext('draw()',sandbox);assert.deepEqual(reads,[0,1,1]);
 console.log(JSON.stringify({status:'PASS_FOR_STATED_TARGET',checks:14,scope:'Real renderer with delayed mocked frame read; queued final-frame drain after pause; no browser or native rerun'}));
})().catch(error=>{console.error(error);process.exitCode=1});
`;
new Function("require", "__dirname", suite + '\n' + regression)(require, __dirname);
