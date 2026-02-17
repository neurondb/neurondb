/**
 * Authentication middleware
 * Auth is required by default when an API key is configured.
 */

import { createHash, timingSafeEqual } from "crypto";
import { Middleware, MCPRequest, MCPResponse } from "../types.js";
import { Logger } from "../../logging/logger.js";

/** Constant-time comparison via SHA-256 hashes to prevent timing attacks on API key validation */
function constantTimeCompare(a: string, b: string): boolean {
	const hashA = createHash("sha256").update(a, "utf8").digest();
	const hashB = createHash("sha256").update(b, "utf8").digest();
	if (hashA.length !== hashB.length) return false;
	return timingSafeEqual(hashA, hashB);
}

export function createAuthMiddleware(apiKey?: string): Middleware {
	return {
		name: "auth",
		order: 0,
		handler: async (request: MCPRequest, next: () => Promise<MCPResponse>): Promise<MCPResponse> => {
			/* When API key is configured, auth is required (reject missing or invalid key) */
			if (apiKey) {
				const requestKey = request.metadata?.apiKey || request.params?.apiKey;
				if (requestKey == null || requestKey === "" || !constantTimeCompare(requestKey, apiKey)) {
					return {
						content: [
							{
								type: "text",
								text: "Unauthorized: Invalid API key",
							},
						],
						isError: true,
					};
				}
			}
			return next();
		},
	};
}





