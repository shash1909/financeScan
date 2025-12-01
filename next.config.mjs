/** @type {import('next').NextConfig} */
const nextConfig = {
  images: {
    remotePatterns: [
      {
        protocol: "https",
        hostname: "randomuser.me",
      },
    ],
  },

  // NEW FORMAT FOR NEXT.JS 15
  serverExternalPackages: [
    "prettier",
    "@react-email/render",
    "@react-email/components",
  ],

  experimental: {
    serverActions: {
      bodySizeLimit: "5mb",
    },
  },
};

export default nextConfig;
