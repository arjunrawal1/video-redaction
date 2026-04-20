import { runPassRoute } from "@/lib/server/run-pass-route";

export const runtime = "nodejs";
export const maxDuration = 600;

export async function POST(req: Request): Promise<Response> {
  return runPassRoute(req, "forward");
}
