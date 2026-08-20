import Image from "next/image";
import Link from "next/link";
import { AccountControl } from "./AccountControl";
import { StreakBadge } from "./StreakBadge";

export function Header() {
  return (
    <header className="border-b-4 border-brand-yellow bg-surface">
      <div className="mx-auto flex max-w-2xl items-center justify-between gap-3 px-6 py-4">
        <Link href="/" className="flex items-center gap-3">
          <Image
            src="/logo.png"
            alt="Saraswati Hindi Jagat logo"
            width={48}
            height={48}
            className="h-12 w-12 rounded-full object-cover ring-4 ring-brand-yellow"
            priority
          />
          <span>
            <span className="block font-[family-name:var(--font-display)] text-xl leading-tight text-brand-pink">
              सरस्वती हिंदी जगत
            </span>
            <span className="block text-xs tracking-wide text-foreground/50">
              Saraswati Hindi Jagat
            </span>
          </span>
        </Link>
        <div className="flex items-center gap-3">
          <AccountControl />
          <StreakBadge />
        </div>
      </div>
    </header>
  );
}
