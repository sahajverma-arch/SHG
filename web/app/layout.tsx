import type { Metadata } from "next";
import { Inter, Noto_Sans_Devanagari, Tiro_Devanagari_Hindi } from "next/font/google";
import { Header } from "@/components/Header";
import "./globals.css";

// Two roles only. Inter carries the Latin interface; Noto Sans Devanagari
// picks up every Devanagari glyph Inter has none of, so the two behave as one
// neutral UI face.
const inter = Inter({
  subsets: ["latin"],
  variable: "--font-inter",
});

const notoDevanagari = Noto_Sans_Devanagari({
  weight: ["400", "500", "600", "700"],
  subsets: ["devanagari", "latin"],
  variable: "--font-noto-dev",
});

// The one characterful face, reserved for the passage itself — cut for
// reading Devanagari rather than for interface text.
const tiro = Tiro_Devanagari_Hindi({
  weight: "400",
  subsets: ["devanagari", "latin"],
  variable: "--font-read",
});

export const metadata: Metadata = {
  title: "सरस्वती हिंदी जगत",
  description: "Daily Hindi reading and vocabulary practice for kids.",
  icons: { icon: "/logo.png" },
};

export const viewport = {
  width: "device-width",
  initialScale: 1,
  viewportFit: "cover" as const,
};

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html
      lang="hi"
      className={`${inter.variable} ${notoDevanagari.variable} ${tiro.variable} h-full`}
    >
      <body className="flex min-h-full flex-col">
        <Header />
        {children}
      </body>
    </html>
  );
}
