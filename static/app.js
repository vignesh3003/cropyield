document.addEventListener("DOMContentLoaded", function () {
  const ajaxBtn = document.getElementById("ajax-btn");
  const form = document.getElementById("predict-form");
  const result = document.getElementById("result");

  ajaxBtn.addEventListener("click", async () => {
    const data = {};
    new FormData(form).forEach((v, k) => (data[k] = v));

    result.innerHTML = '<div class="card">Predicting…</div>';

    try {
      const resp = await fetch("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(data),
      });

      // Batch upload
      const batchBtn = document.getElementById("batch-btn");
      const csvFile = document.getElementById("csv-file");

      batchBtn.addEventListener("click", async () => {
        const f = csvFile.files[0];
        if (!f) {
          result.innerHTML =
            '<div class="card error">Please choose a CSV file.</div>';
          return;
        }

        result.innerHTML = '<div class="card">Uploading and predicting…</div>';

        const fd = new FormData();
        fd.append("file", f);

        try {
          const resp = await fetch("/api/predict_batch", {
            method: "POST",
            body: fd,
          });
          const j = await resp.json();
          if (resp.ok) {
            // show a summary and create a CSV download
            const preds = j.predictions || [];
            let html = `<div class="card"><h3>Batch predictions (${preds.length})</h3>`;
            html +=
              "<ol>" +
              preds
                .map((p) => `<li>row ${p.index}: ${p.prediction}</li>`)
                .join("") +
              "</ol>";
            html += "</div>";
            result.innerHTML = html;
          } else {
            result.innerHTML = `<div class="card error">Error: ${
              j.error || JSON.stringify(j)
            }</div>`;
          }
        } catch (err) {
          result.innerHTML = `<div class="card error">Error: ${err.message}</div>`;
        }
      });
      const j = await resp.json();
      if (resp.ok) {
        result.innerHTML = `<div class="card"><h2>Prediction</h2><p><strong>Predicted yield:</strong> ${j.prediction}</p></div>`;
      } else {
        result.innerHTML = `<div class="card error">Error: ${
          j.error || "Unknown"
        }</div>`;
      }
    } catch (err) {
      result.innerHTML = `<div class="card error">Error: ${err.message}</div>`;
    }
  });
});
// Install prompt handling
let deferredPrompt;
const installBtn = document.getElementById("install-btn");
if (installBtn) installBtn.style.display = "none";

window.addEventListener("beforeinstallprompt", (e) => {
  // Prevent the mini-infobar from appearing on mobile
  e.preventDefault();
  deferredPrompt = e;
  if (installBtn) {
    installBtn.style.display = "inline-block";
    installBtn.addEventListener("click", async () => {
      installBtn.style.display = "none";
      deferredPrompt.prompt();
      const choiceResult = await deferredPrompt.userChoice;
      console.log("User choice", choiceResult);
      deferredPrompt = null;
    });
  }
});

// Register service worker
if ("serviceWorker" in navigator) {
  navigator.serviceWorker
    .register("/static/service-worker.js")
    .then((reg) => console.log("SW registered", reg))
    .catch((err) => console.warn("SW failed", err));
}
