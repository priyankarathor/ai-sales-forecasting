// React tables/dropdowns ke liye monthly MongoDB GET client.
export function createMonthlyDataApi(baseUrl = "http://127.0.0.1:8000") {
  const root = `${baseUrl.replace(/\/$/, "")}/monthly-data`;

  async function get(path, filters = {}) {
    const query = new URLSearchParams();
    Object.entries(filters).forEach(([key, value]) => {
      if (value !== undefined && value !== null && value !== "") query.set(key, String(value));
    });
    const response = await fetch(`${root}${path}${query.size ? `?${query}` : ""}`);
    const body = await response.json();
    if (!response.ok) throw new Error(JSON.stringify(body.detail));
    return body;
  }

  return {
    summary: () => get("/summary"),
    all: (options) => get("/all", options),
    page: (collection, options) => get(`/${encodeURIComponent(collection)}`, options),
    async allRecords(collection, filters = {}) {
      const records = [];
      let skip = 0;
      while (true) {
        const page = await get(`/${encodeURIComponent(collection)}`, { ...filters, skip, limit: 1000 });
        records.push(...page.data);
        if (!page.has_more || page.next_skip === null) break;
        skip = page.next_skip;
      }
      return records;
    },
  };
}
