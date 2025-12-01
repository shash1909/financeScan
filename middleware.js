import { clerkMiddleware } from "@clerk/nextjs/server";
import { NextResponse } from "next/server";
import arcjet, { shield, detectBot } from "@arcjet/next";

// 1. Lightweight function instead of createRouteMatcher
const protectedRoutes = [
  /^\/dashboard/,
  /^\/account/,
  /^\/transaction/,
];

// 2. Create Arcjet instance (keeps size smaller)
const aj = arcjet({
  key: process.env.ARCJET_KEY,
  rules: [
    shield({ mode: "LIVE" }),
    detectBot({
      mode: "LIVE",
      allow: ["CATEGORY:SEARCH_ENGINE", "GO_HTTP"],
    }),
  ],
});

// 3. Clerk + ArcJet combined with minimal logic
export default clerkMiddleware(async (auth, req) => {
  const url = req.nextUrl.pathname;

  // Run Arcjet first
  const ajRes = await aj.protect(req);
  if (ajRes.isDenied()) {
    return NextResponse.json(
      { error: "Request blocked by ArcJet" },
      { status: 403 }
    );
  }

  // Protected routes
  const { userId } = await auth();
  if (!userId && protectedRoutes.some((r) => r.test(url))) {
    const { redirectToSignIn } = await auth();
    return redirectToSignIn();
  }

  return NextResponse.next();
});

// 4. Minimal matcher (smaller = smaller bundle)
export const config = {
  matcher: [
    "/((?!_next|.*\\..*).*)", // only dynamic routes
    "/api/:path*",            // always apply on API
  ],
};

