import type { Metadata } from "next";
import { Hind, Modak } from "next/font/google";
import { Header } from "@/components/Header";
import "./globals.css";

const hind = Hind({
  weight: ["400", "500", "600", "700"],
  subsets: ["devanagari", "latin"],
  variable: "--font-hind",
});

const modak = Modak({
  weight: "400",
  subsets: ["devanagari", "latin"],
  variable: "--font-modak",
});

export const metadata: Metadata = {
  title: "सरस्वती हिंदी जगत",
  description: "Daily Hindi reading and vocabulary practice for kids.",
  icons: { icon: "/logo.png" },
};

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html
      lang="hi"
      className={`${hind.variable} ${modak.variable} h-full antialiased`}
    >
      <body className="min-h-full flex flex-col">
        <Header />
        {children}
      </body>
    </html>
  );
}
