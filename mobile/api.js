import { BASE_URL } from "./config";

async function postPredict(payload) {
  const url = `${BASE_URL}/api/predict`;
  const resp = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  const data = await resp.json();
  if (!resp.ok) throw data;
  return data;
}

export default { postPredict };
