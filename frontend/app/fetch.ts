interface FetchOptions extends RequestInit {
  body?: any;
}
export const fetchUtil = async (url: string, options: FetchOptions = {}) => {
  const { body, ...restOptions } = options;

  const response = await fetch(url, {
    ...restOptions,
    headers: {
      "Content-Type": "application/json",
      ...restOptions.headers,
    },
    body: body ? JSON.stringify(body) : undefined,
  }).catch((error) => {
    console.error("Fetch error:", error);
    throw error;
  });

  if (!response.ok && response.status !== 401) {
    throw new Error(response.statusText);
  }

  return response;
};
