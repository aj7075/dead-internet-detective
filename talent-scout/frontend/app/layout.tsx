import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Talent Scout AI",
  description: "AI-powered recruiter agent — find and rank the best candidates instantly.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="min-h-screen antialiased">{children}</body>
    </html>
  );
}
