import { describe, expect, it } from "vitest";

import { NeuronMCPClient } from "./client.js";

describe("NeuronMCPClient", () => {
  it("constructs with baseUrl", () => {
    const client = new NeuronMCPClient({ baseUrl: "http://localhost:8080/" });
    expect(client).toBeDefined();
  });
});
