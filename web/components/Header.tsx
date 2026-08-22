import Image from "next/image";
import Link from "next/link";
import { AccountControl } from "./AccountControl";
import { StreakBadge } from "./StreakBadge";

export function Header() {
  return (
    <header className="sticky top-0 z-40 bg-surface/90 shadow-[0_1px_0_rgba(0,0,0,0.04),0_12px_24px_-18px_rgba(0,0,0,0.4)] backdrop-blur-sm">
      <div className="mx-auto flex max-w-3xl items-center justify-between gap-3 px-6 py-3.5">
        <Link href="/" className="group flex items-center gap-3">
          <Image
            src="/logo.png"
            alt="Saraswati Hindi Jagat logo"
            width={44}
            height={44}
            className="h-11 w-11 rounded-full object-cover ring-4 ring-brand-yellow transition-transform duration-200 group-hover:-rotate-6 group-hover:scale-105"
            priority
          />
          <span>
            <span className="block font-[family-name:var(--font-display)] text-lg leading-tight text-brand-pink">
              सरस्वती हिंदी जगत
            </span>
            <span className="block text-[11px] tracking-wide text-foreground/45">
              Saraswati Hindi Jagat
            </span>
          </span>
        </Link>
        <div className="flex items-center gap-3">
          <AccountControl />
          <StreakBadge />
        </div>
      </div>
      <div className="h-1 w-full bg-[linear-gradient(90deg,var(--brand-pink),var(--brand-orange),var(--brand-yellow),var(--brand-green),var(--brand-blue))]" />
    </header>
  );
}
