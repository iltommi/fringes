let pyodide;

async function loadPyodideAndPackages() {
  document.getElementById("loadingBanner").style.display = "block";

  pyodide = await loadPyodide();
  await pyodide.loadPackage(["numpy", "matplotlib", "pillow"]);

  const mainCode = await (await fetch("main.py")).text();
  await pyodide.runPythonAsync(mainCode);

  document.getElementById("loadingBanner").style.display = "none";
}

loadPyodideAndPackages();

async function runAnalysis() {
  const banner = document.getElementById("loadingBanner");
  banner.textContent = "Running analysis, please wait...";
  banner.style.display = "block";

  try {
    const fileRef = document.getElementById("fileRef").files[0];
    const fileShot = document.getElementById("fileShot").files[0];

    const readAsBytes = (file) => new Promise((resolve) => {
      const reader = new FileReader();
      reader.onload = () => resolve(reader.result);
      reader.readAsArrayBuffer(file);
    });

    const [refBuf, shotBuf] = await Promise.all([
      readAsBytes(fileRef),
      readAsBytes(fileShot),
    ]);

    pyodide.FS.writeFile("ref.tiff", new Uint8Array(refBuf));
    pyodide.FS.writeFile("shot.tiff", new Uint8Array(shotBuf));

    const wl = parseInt(document.getElementById("wlSteps").value);
    const al = parseInt(document.getElementById("alSteps").value);
    const cutoff = parseFloat(document.getElementById("paramCutoff").value);

    let output = "";
    pyodide.setStdout({
      batched: (text) => output += text + "\n",
    });

    const pythonCode = `analyze('ref.tiff', 'shot.tiff', wl=${wl}, al=${al}, cutoff=${cutoff})`;
    await pyodide.runPythonAsync(pythonCode);

    document.getElementById("output").textContent = output;

    const outputImage = pyodide.FS.readFile("/output.png", { encoding: "binary" });
    const blob = new Blob([outputImage], { type: "image/png" });
    const url = URL.createObjectURL(blob);
    document.getElementById("resultImg").src = url;

    const outputTiff = pyodide.FS.readFile("/output.tiff", { encoding: "binary" });
    const tiffBlob = new Blob([outputTiff], { type: "image/tiff" });
    const tiffUrl = URL.createObjectURL(tiffBlob);
    const downloadLink = document.getElementById("downloadLink");
    downloadLink.href = tiffUrl;
    downloadLink.style.display = "inline";
  } catch (error) {
    document.getElementById("output").textContent = "Error: " + error.message;
  } finally {
    banner.style.display = "none";
  }
}
// 
// async function runAnalysis() {
//   const fileRef = document.getElementById("fileRef").files[0];
//   const fileShot = document.getElementById("fileShot").files[0];
// 
//   const readAsBytes = (file) => new Promise((resolve) => {
//     const reader = new FileReader();
//     reader.onload = () => resolve(reader.result);
//     reader.readAsArrayBuffer(file);
//   });
// 
//   const [refBuf, shotBuf] = await Promise.all([
//     readAsBytes(fileRef),
//     readAsBytes(fileShot),
//   ]);
// 
//   pyodide.FS.writeFile("ref.tiff", new Uint8Array(refBuf));
//   pyodide.FS.writeFile("shot.tiff", new Uint8Array(shotBuf));
// 
//   const wl = parseInt(document.getElementById("wlSteps").value);
//   const al = parseInt(document.getElementById("alSteps").value);
//   const cutoff = parseFloat(document.getElementById("paramCutoff").value);
// 
//   // Redirect stdout to capture printed output
//   let output = "";
//   pyodide.setStdout({
//     batched: (text) => output += text + "\n",
//   });
// 
//   const pythonCode = `analyze('ref.tiff', 'shot.tiff', wl=${wl}, al=${al}, cutoff=${cutoff})`;
//   await pyodide.runPythonAsync(pythonCode);
// 
//   // Show printed output in HTML
//   document.getElementById("output").textContent = output;
// 
//   const outputImage = pyodide.FS.readFile("/output.png", { encoding: "binary" });
//   const blob = new Blob([outputImage], { type: "image/png" });
//   const url = URL.createObjectURL(blob);
//   document.getElementById("resultImg").src = url;
// 
//   const outputTiff = pyodide.FS.readFile("/output.tiff", { encoding: "binary" });
//   const tiffBlob = new Blob([outputTiff], { type: "image/tiff" });
//   const tiffUrl = URL.createObjectURL(tiffBlob);
//   const downloadLink = document.getElementById("downloadLink");
//   downloadLink.href = tiffUrl;
//   downloadLink.style.display = "inline";
// }
