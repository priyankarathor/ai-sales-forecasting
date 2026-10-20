// React controls/charts se ye methods call karein.
export function createMonthlyMlApi(baseUrl = "http://127.0.0.1:8000") {
  const root = `${baseUrl.replace(/\/$/, "")}/predictions/monthly`;

  async function request(path, { method = "GET", body } = {}) {
    const response = await fetch(`${root}${path}`, {
      method,
      ...(body === undefined ? {} : {
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      }),
    });
    const data = await response.json();
    if (!response.ok) {
      throw new Error(typeof data.detail === "string" ? data.detail : JSON.stringify(data.detail));
    }
    return data;
  }

  function query(filters = {}) {
    const params = new URLSearchParams();
    Object.entries(filters).forEach(([key, value]) => {
      if (value !== undefined && value !== null && value !== "") params.set(key, String(value));
    });
    return params.size ? `?${params}` : "";
  }

  return {
    details: () => request("/details"),
    history: (product_id, country_id) => request(`/history${query({ product_id, country_id })}`),
    evaluation: (filters) => request(`/evaluation${query(filters)}`),
    forecasts: (filters) => request(query(filters)),
    predict: (body) => request("/predict", { method: "POST", body }),
    // Training complete hone mein lagbhag 1-2 minutes lag sakte hain.
    train: () => request("/train", { method: "POST" }),
  };
}
