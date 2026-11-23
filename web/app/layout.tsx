import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Concrete Strength Predictor",
  description:
    "Estimate compressive strength in MPa from eight mixture proportions. An estimate, not a substitute for laboratory testing.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
