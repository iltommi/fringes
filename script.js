let pyodide;

async function loadPyodideAndPackages() {
  pyodide = await loadPyodide();
  await pyodide.loadPackage(["numpy", "matplotlib", "pillow", "scipy"]);

  const mainCode = await (await fetch("main.py")).text();
  await pyodide.runPythonAsync(mainCode);

}

loadPyodideAndPackages();

async function runAnalysis() {
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

  const pythonCode = `analyze('ref.tiff', 'shot.tiff', wl=${wl}, al=${al}, cutoff=${cutoff})`;

  await pyodide.runPythonAsync(pythonCode);

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
}
