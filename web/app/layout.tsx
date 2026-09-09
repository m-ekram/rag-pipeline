import type { Metadata } from "next";
import {
  Instrument_Serif,
  JetBrains_Mono,
  Noto_Sans_Devanagari,
  Public_Sans,
} from "next/font/google";
import "./globals.css";

const publicSans = Public_Sans({
  subsets: ["latin"],
  variable: "--font-public-sans",
  display: "swap",
});

const instrumentSerif = Instrument_Serif({
  subsets: ["latin"],
  weight: "400",
  variable: "--font-instrument-serif",
  display: "swap",
});

const jetbrainsMono = JetBrains_Mono({
  subsets: ["latin"],
  variable: "--font-jetbrains-mono",
  display: "swap",
});

// Loaded up front rather than fallen back to: the corpus is Devanagari, and
// system fallbacks render Hindi differently on every machine.
const notoDevanagari = Noto_Sans_Devanagari({
  subsets: ["devanagari"],
  variable: "--font-noto-deva",
  display: "swap",
});

export const metadata: Metadata = {
  title: "Sanchay — grounded document retrieval",
  description:
    "Ask questions of a folder of scanned documents. Every answer cites its pages, and the system abstains when the evidence is not there.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" suppressHydrationWarning>
      <head>
        {/* Applied before paint so the first frame is already in the right
            theme — a flash of the wrong ground is the most visible bug a
            themed app can ship. */}
        <script
          dangerouslySetInnerHTML={{
            __html: `(function(){try{var t=localStorage.getItem("theme");if(t)document.documentElement.setAttribute("data-theme",t);}catch(e){}})();`,
          }}
        />
      </head>
      <body
        className={`${publicSans.variable} ${instrumentSerif.variable} ${jetbrainsMono.variable} ${notoDevanagari.variable}`}
      >
        {children}
      </body>
    </html>
  );
}
