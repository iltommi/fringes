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
  
  const packages = ["numpy", "pillow"];
  
  for (const pkg of packages) {
    showBanner(`Loading Pyodide ${pkg}`);
    await pyodide.loadPackage(pkg);
  }
  
  showBanner(`Loading Pyodide plotly`);
  await pyodide.loadPackage("micropip");
  const micropip = pyodide.pyimport("micropip");
  await micropip.install('plotly');

//   await pyodide.loadPackage(["numpy", "matplotlib", "plotly", "pillow", "scipy"]);

  showBanner(`Loading Pyodide main module`);
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

  try {
    const fileRef = document.getElementById("fileRef").files[0];
    const fileShot = document.getElementById("fileShot").files[0];
    if (fileRef && fileShot) {
    
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
    }

    const wl = parseInt(document.getElementById("wlSteps").value);
    const al = parseInt(document.getElementById("alSteps").value);
    const cutoff = parseFloat(document.getElementById("paramCutoff").value);

    let output = "";
    pyodide.setStdout({
      batched: (text) => output += text + "\n",
    });

    const pythonCode = `analyze('ref.tiff', 'shot.tiff', wl=${wl}, al=${al}, cutoff=${cutoff})`;
    await pyodide.runPythonAsync(pythonCode);
    console.log(output);

    const resultHtml = pyodide.FS.readFile("/output.html", { encoding: "binary" });
    const htmlBlob = new Blob([resultHtml], { type: "text/html" });
    const htmlUrl = URL.createObjectURL(htmlBlob);
    if (oldImageUrl) URL.revokeObjectURL(oldImageUrl);
    oldImageUrl = htmlUrl;
    document.getElementById("resultFrame").src = oldImageUrl;

    const tiffData = pyodide.FS.readFile("/output.tiff", { encoding: "binary" });
    const tiffBlob = new Blob([tiffData], { type: "image/tiff" });
    const tiffUrl = URL.createObjectURL(tiffBlob);
    if (oldTiffUrl) URL.revokeObjectURL(oldTiffUrl);
    oldTiffUrl = tiffUrl;

    const downloadLink = document.getElementById("downloadLink");
    downloadLink.href = oldTiffUrl;
    downloadLink.classList.remove("hidden");
  } catch (error) {
    console.log(error.message);
  } finally {
    hideBanner();
    toggleForm(false);
  }
}
