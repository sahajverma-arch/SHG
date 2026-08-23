import Image from "next/image";
import Link from "next/link";
import { AccountControl } from "./AccountControl";
import { StreakBadge } from "./StreakBadge";

export function Header() {
  return (
    <header className="sticky top-0 z-40 border-b border-border bg-background/85 backdrop-blur">
      <div className="mx-auto flex max-w-2xl items-center justify-between gap-3 px-4 py-2.5 sm:px-6">
        <Link href="/" className="flex min-w-0 items-center gap-2.5">
          <Image
            src="/logo.png"
            alt=""
            width={32}
            height={32}
            className="h-8 w-8 shrink-0 rounded-full object-cover"
            priority
          />
          <span className="truncate text-[0.9375rem] font-semibold">
            सरस्वती हिंदी जगत
          </span>
        </Link>
        <div className="flex shrink-0 items-center gap-2">
          <AccountControl />
          <StreakBadge />
        </div>
      </div>
    </header>
  );
}
