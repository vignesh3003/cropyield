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
