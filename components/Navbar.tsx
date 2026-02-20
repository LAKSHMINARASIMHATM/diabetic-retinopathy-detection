"use client"

import Link from "next/link"
import { usePathname } from "next/navigation"
import { Microscope, Eye, ClipboardList, Info, Activity, FileText } from "lucide-react"

const NAV_LINKS = [
    { href: "/", label: "Home", icon: Eye },
    { href: "/analyze", label: "Image Analysis", icon: Activity },
    { href: "/clinical", label: "Clinical Assessment", icon: ClipboardList },
    { href: "/report", label: "Prescription", icon: FileText },
    { href: "/about", label: "About", icon: Info },
]

export function Navbar() {
    const pathname = usePathname()

    return (
        <header style={{
            position: "sticky", top: 0, zIndex: 50,
            borderBottom: "1px solid rgba(129,140,248,0.12)",
            backdropFilter: "blur(16px)",
            background: "rgba(8,13,26,0.85)",
        }}>
            <div style={{
                maxWidth: 1200, margin: "0 auto",
                padding: "0 24px",
                display: "flex", alignItems: "center",
                justifyContent: "space-between",
                height: 64,
            }}>
                {/* Logo */}
                <Link href="/" style={{ display: "flex", alignItems: "center", gap: 12, textDecoration: "none" }}>
                    <div style={{
                        width: 38, height: 38, borderRadius: 11,
                        background: "linear-gradient(135deg, #818cf8 0%, #6366f1 100%)",
                        display: "flex", alignItems: "center", justifyContent: "center",
                        boxShadow: "0 0 18px rgba(129,140,248,0.35)",
                    }}>
                        <Microscope size={20} color="#fff" />
                    </div>
                    <div>
                        <div style={{ fontWeight: 800, fontSize: 16, color: "#f1f5f9", letterSpacing: "-0.02em" }}>RetinaAI</div>
                        <div style={{ fontSize: 10, color: "#475569", letterSpacing: "0.06em", textTransform: "uppercase" }}>
                            DR Detection System
                        </div>
                    </div>
                </Link>

                {/* Nav links */}
                <nav style={{ display: "flex", alignItems: "center", gap: 4 }}>
                    {NAV_LINKS.map(({ href, label, icon: Icon }) => {
                        const active = pathname === href
                        return (
                            <Link
                                key={href}
                                href={href}
                                style={{
                                    display: "flex", alignItems: "center", gap: 7,
                                    padding: "7px 14px", borderRadius: 10,
                                    fontSize: 13, fontWeight: active ? 600 : 500,
                                    textDecoration: "none",
                                    color: active ? "#818cf8" : "#64748b",
                                    background: active ? "rgba(129,140,248,0.12)" : "transparent",
                                    border: active ? "1px solid rgba(129,140,248,0.2)" : "1px solid transparent",
                                    transition: "all 0.2s",
                                }}
                                onMouseEnter={e => {
                                    if (!active) {
                                        (e.currentTarget as HTMLAnchorElement).style.color = "#94a3b8"
                                            ; (e.currentTarget as HTMLAnchorElement).style.background = "rgba(255,255,255,0.05)"
                                    }
                                }}
                                onMouseLeave={e => {
                                    if (!active) {
                                        (e.currentTarget as HTMLAnchorElement).style.color = "#64748b"
                                            ; (e.currentTarget as HTMLAnchorElement).style.background = "transparent"
                                    }
                                }}
                            >
                                <Icon size={14} />
                                {label}
                            </Link>
                        )
                    })}
                </nav>

                {/* Status dot */}
                <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                    <div style={{
                        display: "flex", alignItems: "center", gap: 7,
                        padding: "6px 14px", borderRadius: 100,
                        background: "rgba(34,197,94,0.08)",
                        border: "1px solid rgba(34,197,94,0.2)",
                        fontSize: 11, fontWeight: 600, color: "#22c55e",
                    }}>
                        <div style={{
                            width: 6, height: 6, borderRadius: "50%",
                            background: "#22c55e",
                            boxShadow: "0 0 6px #22c55e",
                            animation: "pulse-glow 2.5s ease-in-out infinite",
                        }} />
                        Real Model
                    </div>
                </div>
            </div>
        </header>
    )
}
