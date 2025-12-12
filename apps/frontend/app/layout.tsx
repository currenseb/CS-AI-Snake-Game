import { ThemeProvider } from "@/components/theme-provider";
import type { Metadata } from "next";
import { Header } from "../components/Header";
import "./globals.css";

export const metadata: Metadata = {
  title: "CS + AI Snake Bootcamp",
  description: "CSAI Fall 2025 Snake Bootcamp Project",
};

// Fixed root layout with mobile-safe height + scrolling
export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" suppressHydrationWarning>
      <head />
      <body className="min-h-[100svh] flex flex-col">
        <ThemeProvider
          attribute="class"
          defaultTheme="system"
          enableSystem
          disableTransitionOnChange
        >
          {/* Sticky header takes natural height */}
          <header className="sticky top-0 z-50">
            <Header />
          </header>

          {/* Main content gets remaining space and can scroll */}
          <main className="flex-1 overflow-y-auto">
            {children}
          </main>
        </ThemeProvider>
      </body>
    </html>
  );
}
