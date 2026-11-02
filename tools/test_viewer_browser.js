"use strict";
/** Actual Chromium interaction checks for a finalized native dataset.
 * Usage: node tools/test_viewer_browser.js URL RUN_DIRECTORY OUTPUT_DIRECTORY [CHROME]
 * Needs the optional Playwright package and a running loopback serve_viewer.py.
 * Screenshots and a JSON check record are retained; native files are not changed.
 */
const {chromium} = require("playwright");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const [url, run, output, executablePath] = process.argv.slice(2);
if (!url || !run || !output) throw Error("Supply URL RUN_DIRECTORY OUTPUT_DIRECTORY [CHROME]");
const target = new URL(url);
assert(["127.0.0.1", "localhost", "[::1]"].includes(target.hostname));
fs.mkdirSync(output, {recursive:true});
(async () => {
    const browser = await chromium.launch({headless:true, ...(executablePath ? {executablePath} : {})});
    const page = await browser.newPage({viewport:{width:1024,height:900},colorScheme:"light"});
    const errors=[]; page.on("pageerror",e=>errors.push(e.message));
    const checks=[];
    try {
        await page.goto(url);
        await page.waitForFunction(()=>!document.getElementById("app").hidden,{timeout:120000});
        const frame = page.locator("#fr");
        await frame.fill("3"); await frame.dispatchEvent("input");
        await frame.focus(); await page.keyboard.press("ArrowRight");
        assert.equal(await frame.inputValue(),"4"); checks.push("native slider advances once");
        await page.locator("#scopeOverrides").check();
        await page.locator("#spectrumScope").focus(); await page.keyboard.press("ArrowRight");
        assert.equal(await frame.inputValue(),"4"); checks.push("focused select leaves frame unchanged");
        await page.locator("#scopeOverrides").uncheck();
        await page.locator("#globalScope").selectOption("frame");
        for (const id of ["spectrumScope","distributionScope","secondaryScope","polScope","diagScope","momentScope"])
            assert.equal(await page.locator("#"+id).inputValue(),"frame");
        checks.push("global scope and per-panel badges");
        await page.locator("#c").click(); await page.keyboard.press("ArrowRight");
        assert.equal(await frame.inputValue(),"5"); checks.push("page shortcut advances once");
        assert(await page.locator("#openAnother").isVisible());
        await page.locator("#dir").setInputFiles(path.resolve(run));
        await page.waitForFunction(()=>!document.getElementById("openAnother").disabled);
        assert(await page.locator("#app").isVisible()); checks.push("visible chooser opens valid replacement");
        const identity = await page.locator("#datasetIdentity").textContent();
        const invalid = path.join(output,"invalid_dataset_fixture");fs.mkdirSync(invalid,{recursive:true});
        fs.writeFileSync(path.join(invalid,"gui_manifest.json"),"{}");
        await page.locator("#dir").setInputFiles(path.resolve(invalid));
        await page.waitForFunction(()=>document.getElementById("status").textContent.includes("previous run retained"),{timeout:120000});
        assert.equal(await page.locator("#datasetIdentity").textContent(),identity);
        checks.push("invalid replacement preserves previous dataset");
        await page.locator("#dir").setInputFiles(path.resolve(run));
        await page.waitForFunction(()=>!document.getElementById("openAnother").disabled);
        await frame.fill("7"); await frame.dispatchEvent("input");
        if (await page.locator("#fieldMode").isEnabled()) await page.locator("#fieldMode").selectOption("on");
        for (const theme of ["light","dark"]) {
            await page.evaluate(theme=>document.documentElement.dataset.theme=theme,theme);
            await page.evaluate(()=>draw());
            for (const width of [1024,375]) {
                await page.setViewportSize({width,height:900});
                await page.evaluate(()=>draw());
                assert(await page.evaluate(()=>document.documentElement.scrollWidth <= innerWidth+1));
                await page.locator("#c").scrollIntoViewIfNeeded();
                await page.screenshot({path:path.join(output,`viewer_${theme}_${width}.png`)});
            }
        }
        checks.push("light/dark layouts at 375 and 1024 pixels");
        assert.deepEqual(errors,[]);
        fs.writeFileSync(path.join(output,"browser_checks.json"),JSON.stringify({status:"PASS",browser:await browser.version(),checks},null,2));
        console.log(JSON.stringify({status:"PASS",checks}));
    } finally { await browser.close(); }
})().catch(e=>{console.error(e);process.exitCode=1;});
