"use strict";
/** Validate a finalized dataset with the browser's exact scientific contracts.
 * Usage: node tools/validate_viewer_dataset.js RUN_DIRECTORY [...]
 * Reads local files only; returns nonzero on hash, bounds or scientific closure errors.
 */
const fs = require("node:fs"), path = require("node:path");
if (!globalThis.crypto) globalThis.crypto = require("node:crypto").webcrypto;
const contract = require("./collider_contract.js");
const fields = require("./collider_fields.js");
const mono = require("./collider_monochromatization.js");
const secondary = require("./collider_secondary_kinematics.js");
const events = require("./collider_event_deflections.js");
async function validate(directory) {
    const root = path.resolve(directory), read = name => {
        contract.safeName(name);
        const file = path.join(root,name);
        if (fs.lstatSync(file).isSymbolicLink()) throw Error("Linked dataset artifact");
        return fs.readFileSync(file);
    };
    const manifest = contract.manifest(JSON.parse(read("gui_manifest.json")));
    for (const frame of manifest.frames) for (const key of ["beam1","beam2"])
        await contract.artifact(read(frame[key].path),frame[key]);
    const refs = [...(manifest.analysis_sources||[]),...(manifest.diagnostic_analysis_sources||[])];
    for (const key of ["analysis","analysis_csv","diagnostics","fields","monochromatization","secondary_kinematics","event_deflections"])
        if (manifest[key]) refs.push(manifest[key]);
    const bytes = new Map();
    for (const ref of refs) {
        if (bytes.has(ref.path)) continue;
        const data = read(ref.path);
        await contract.bytesHash(data,ref,128*1024*1024,true);
        bytes.set(ref.path,data);
    }
    const json = ref => ref ? JSON.parse(bytes.get(ref.path)) : null;
    const analysis = manifest.analysis ? contract.analysis(json(manifest.analysis)) : null;
    const diagnostics = manifest.diagnostics ? contract.diagnostics(json(manifest.diagnostics),manifest) : null;
    if (diagnostics) contract.bindPeakReference(diagnostics,manifest,analysis);
    if (manifest.fields) fields.validate(json(manifest.fields),manifest);
    if (manifest.monochromatization) mono.validate(json(manifest.monochromatization),refs,{manifest,analysis,diagnostics});
    const optional = name => bytes.has(name) ? JSON.parse(bytes.get(name)) : null;
    if (manifest.secondary_kinematics) secondary.validate(json(manifest.secondary_kinematics),refs,
        {manifest,frameSnapshots:optional("gui_secondary_frames.json")});
    if (manifest.event_deflections) events.validate(json(manifest.event_deflections),refs,
        {manifest,trajectories:optional("gui_imported_trajectories.json"),identity:optional("event_identity.json"),
         inputText:bytes.get("event.ini")?.toString("utf8"),summaryText:bytes.get("out.dat")?.toString("utf8")});
    return {directory:root,status:"PASS_BROWSER_DATA_CONTRACT",frames:manifest.frames.length,
        physics_qualification:"NOT_ESTABLISHED"};
}
(async()=>{
    if (!process.argv.slice(2).length) throw Error("Supply dataset directories");
    for (const directory of process.argv.slice(2)) console.log(JSON.stringify(await validate(directory)));
})().catch(error=>{console.error(error);process.exitCode=1;});
