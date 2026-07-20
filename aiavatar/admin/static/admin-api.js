const apiBase = new URL("api/", document.baseURI);

async function request(path, options = {}) {
  const response = await fetch(new URL(path.replace(/^\//, ""), apiBase), {
    ...options,
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
  });
  if (!response.ok) {
    let message = `${response.status} ${response.statusText}`;
    try {
      const body = await response.json();
      message = body.detail || message;
    } catch (_) {}
    throw new Error(message);
  }
  const type = response.headers.get("content-type") || "";
  return type.includes("application/json") ? response.json() : response;
}

export const api = {
  get(path) { return request(path); },
  post(path, body) { return request(path, { method: "POST", body: JSON.stringify(body) }); },
  url(path) { return new URL(path.replace(/^\//, ""), apiBase).toString(); },
};
