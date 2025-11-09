"use client";

import { Button } from "./ui/button";
import Link from "next/link";

export const Header = () => {
  return (
    <header className="flex items-center justify-between h-full px-6 py-4 bg-card border-b border-primary border-b-[1px] relative">
      {/* Left side — LinkedIn button + text */}
      <div className="flex items-center justify-start gap-3 whitespace-nowrap">
        <Button
          asChild
          className="bg-blue-600 text-white hover:bg-blue-700 transition"
        >
          <Link
            href="https://www.linkedin.com/in/currensebastian/"
            target="_blank"
            rel="noopener noreferrer"
          >
            LinkedIn
          </Link>
        </Button>
        <span className="text-lg text-gray-300">← Connect With Me!</span>
      </div>

      {/* Centered Title */}
      <div className="flex-1 text-center">
        <span className="text-xl md:text-2xl font-bold text-primary tracking-wide">
            Cal Poly CS + AI Snake By Curren Sebastian
        </span>
      </div>

      {/* Right side — More of my work + GitHub */}
      <div className="flex items-center justify-end gap-3 whitespace-nowrap">
        <span className="text-lg text-gray-300">More Of My Work →</span>
        <Button
          asChild
          className="bg-primary text-white hover:bg-primary/80 transition"
        >
          <Link
            href="https://github.com/currenseb"
            target="_blank"
            rel="noopener noreferrer"
          >
            GitHub
          </Link>
        </Button>
      </div>
    </header>
  );
};
