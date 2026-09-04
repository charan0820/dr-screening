"use strict";

const { ReplitConnectors } = require("@replit/connectors-sdk");

async function main() {
  let payload;
  try {
    payload = JSON.parse(await readStdin());
  } catch {
    throw new Error("The email payload was not valid JSON.");
  }

  const connectors = new ReplitConnectors();
  const response = await connectors.proxy("resend", "/emails", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: payload,
  });
  const rawBody = await response.text();
  let body = rawBody;
  try {
    body = JSON.parse(rawBody);
  } catch {
    // Preserve a useful provider error when the response is not JSON.
  }

  process.stdout.write(
    JSON.stringify({
      ok: response.ok,
      status: response.status,
      body,
    }),
  );
}

function readStdin() {
  return new Promise((resolve, reject) => {
    let input = "";
    process.stdin.setEncoding("utf8");
    process.stdin.on("data", (chunk) => {
      input += chunk;
    });
    process.stdin.on("end", () => resolve(input));
    process.stdin.on("error", reject);
  });
}

main().catch((error) => {
  process.stderr.write(error instanceof Error ? error.message : String(error));
  process.exitCode = 1;
});