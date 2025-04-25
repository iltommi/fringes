let pyodide;
let oldImageUrl = null;
let oldTiffUrl = null;

function showBanner(message) {
  const banner = document.getElementById("loadingBanner");
  const loadingText = document.getElementById("loadingText");
  loadingText.textContent = message;
  banner.classList.add("show");
}

function hideBanner() {
  const banner = document.getElementById("loadingBanner");
  banner.classList.remove("show");
}

function toggleForm(disabled) {
  document.querySelectorAll("#analysisForm input, #analysisForm button").forEach(el => {
    el.disabled = disabled;
  });
}

async function loadPyodideAndPackages() {
  showBanner("Loading Pyodide packages, please wait...");
  toggleForm(true);

  pyodide = await loadPyodide();
  await pyodide.loadPackage(["numpy", "matplotlib", "pillow", "scipy"]);

  const mainCode = await (await fetch("main.py")).text();
  await pyodide.runPythonAsync(mainCode);

  hideBanner();
  toggleForm(false);
  document.querySelector("button[type='submit']").disabled = false;
}

loadPyodideAndPackages();

async function runAnalysis() {
  showBanner("Running analysis, please wait...");
  toggleForm(true);

  const outputDiv = document.getElementById("output");
  outputDiv.textContent = "";
  document.getElementById("resultImg").src = ""; // <-- clear previous image

  try {
    const fileRef = document.getElementById("fileRef").files[0];
    const fileShot = document.getElementById("fileShot").files[0];
    if (!fileRef || !fileShot) throw new Error("Both files must be selected.");

    const readAsBytes = (file) => new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => resolve(reader.result);
      reader.onerror = () => reject("Failed to read file");
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
    outputDiv.textContent = output;

    const resultImg = document.getElementById("resultImg");
    const resultPng = pyodide.FS.readFile("/output.png", { encoding: "binary" });
    const blob = new Blob([resultPng], { type: "image/png" });
    if (oldImageUrl) URL.revokeObjectURL(oldImageUrl);
    oldImageUrl = URL.createObjectURL(blob);
    resultImg.src = oldImageUrl;

    const tiffData = pyodide.FS.readFile("/output.tiff", { encoding: "binary" });
    const tiffBlob = new Blob([tiffData], { type: "image/tiff" });
    const tiffUrl = URL.createObjectURL(tiffBlob);
    if (oldTiffUrl) URL.revokeObjectURL(oldTiffUrl);
    oldTiffUrl = tiffUrl;

    const downloadLink = document.getElementById("downloadLink");
    downloadLink.href = oldTiffUrl;
    downloadLink.classList.remove("hidden");
  } catch (error) {
    outputDiv.innerHTML = `<div class='alert alert-danger'>Error: ${error.message}</div>`;
  } finally {
    hideBanner();
    toggleForm(false);
  }
}
