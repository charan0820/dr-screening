const { ReplitConnectors } = require("@replit/connectors-sdk");

async function main() {
  const input = await new Promise((resolve, reject) => {
    let data = "";
    process.stdin.setEncoding("utf8");
    process.stdin.on("data", (chunk) => {
      data += chunk;
    });
    process.stdin.on("end", () => resolve(data));
    process.stdin.on("error", reject);
  });

  const payload = JSON.parse(input);
  const connectors = new ReplitConnectors();
  const response = await connectors.proxy("resend", "/emails", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: {
      from: payload.from,
      to: [payload.to],
      subject: payload.subject,
      text: payload.text,
      attachments: [
        {
          filename: payload.filename,
          content: payload.pdf_base64,
        },
      ],
    },
  });

  const responseText = await response.text();
  let responseBody = {};
  try {
    responseBody = responseText ? JSON.parse(responseText) : {};
  } catch {
    responseBody = { message: responseText };
  }

  if (!response.ok) {
    console.error(
      JSON.stringify({
        ok: false,
        status: response.status,
        message: responseBody.message || responseBody.name || "Resend rejected the email",
      }),
    );
    process.exit(1);
  }

  process.stdout.write(JSON.stringify({ ok: true, id: responseBody.id || null }));
}

main().catch((error) => {
  console.error(JSON.stringify({ ok: false, message: error.message }));
  process.exit(1);
});