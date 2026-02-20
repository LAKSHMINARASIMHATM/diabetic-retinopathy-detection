"use client"

import Link from "next/link"
import {
  Eye, Brain, Shield, Activity, Microscope, ArrowRight,
  ClipboardList, Zap, BarChart3, CheckCircle, AlertTriangle
} from "lucide-react"
import { Navbar } from "@/components/Navbar"

const FEATURES = [
  {
    icon: Eye,
    title: "Fundus Image Analysis",
    desc: "Upload a retinal fundus photograph and get instant DR grading powered by EfficientNetB3 trained on 250k+ clinical images.",
    color: "#818cf8",
    href: "/analyze",
    cta: "Analyze Image",
  },
  {
    icon: ClipboardList,
    title: "Clinical Assessment",
    desc: "Enter patient clinical data (HbA1c, BP, VA, demographics) and get DR risk prediction using a validated Random Forest model.",
    color: "#34d399",
    href: "/clinical",
    cta: "Start Assessment",
  },
]

const STATS = [
  { icon: Eye, label: "Fundus Images Trained", value: "250k+", color: "#818cf8" },
  { icon: Brain, label: "Model Architecture", value: "EffNetB3", color: "#34d399" },
  { icon: Activity, label: "DR Severity Levels", value: "5", color: "#f59e0b" },
  { icon: Shield, label: "Clinical Features", value: "24", color: "#f87171" },
]

const DR_LEVELS = [
  { label: "No DR", grade: 0, color: "#22c55e", desc: "No visible signs of retinopathy." },
  { label: "Mild DR", grade: 1, color: "#84cc16", desc: "Microaneurysms only." },
  { label: "Moderate DR", grade: 2, color: "#f59e0b", desc: "More than mild. Multiple lesions." },
  { label: "Severe DR", grade: 3, color: "#ef4444", desc: "Significant neovascularization risk." },
  { label: "Proliferative DR", grade: 4, color: "#dc2626", desc: "New vessel growth. Immediate care needed." },
]

export default function HomePage() {
  return (
    <div style={{ minHeight: "100vh", color: "#e2e8f0", fontFamily: "'Inter', sans-serif" }}>
      <Navbar />

      <main style={{ maxWidth: 1200, margin: "0 auto", padding: "72px 24px 80px" }}>

        {/* ── Hero ── */}
        <div style={{ textAlign: "center", marginBottom: 72 }} className="animate-fade-up">
          <div style={{
            display: "inline-flex", alignItems: "center", gap: 8,
            background: "rgba(129,140,248,0.1)", border: "1px solid rgba(129,140,248,0.25)",
            borderRadius: 100, padding: "7px 18px", marginBottom: 28,
          }}>
            <div style={{ width: 7, height: 7, borderRadius: "50%", background: "#818cf8", boxShadow: "0 0 10px #818cf8" }} />
            <span style={{ fontSize: 12, color: "#818cf8", fontWeight: 700, letterSpacing: "0.07em" }}>
              REAL AI MODEL · NO DEMO DATA
            </span>
          </div>

          <h1 style={{
            fontSize: "clamp(36px,5.5vw,60px)", fontWeight: 900,
            lineHeight: 1.08, letterSpacing: "-0.04em", marginBottom: 20,
            background: "linear-gradient(135deg, #f1f5f9 0%, #a5b4fc 50%, #818cf8 100%)",
            WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent", backgroundClip: "text",
          }}>
            Diabetic Retinopathy<br />Detection System
          </h1>

          <p style={{ fontSize: 18, color: "#64748b", maxWidth: 560, margin: "0 auto", lineHeight: 1.75 }}>
            Dual AI engine for 5-level DR severity grading — fundus image deep learning
            or clinical feature-based prediction. Built for real clinical workflows.
          </p>

          <div style={{ display: "flex", gap: 14, justifyContent: "center", marginTop: 38, flexWrap: "wrap" }}>
            <Link href="/analyze" style={{
              display: "inline-flex", alignItems: "center", gap: 8,
              padding: "13px 30px", borderRadius: 12,
              background: "linear-gradient(135deg, #818cf8, #6366f1)",
              color: "#fff", fontWeight: 700, fontSize: 14,
              textDecoration: "none", transition: "all 0.2s",
              boxShadow: "0 0 28px rgba(99,102,241,0.4)",
            }}>
              <Eye size={16} /> Start Image Analysis <ArrowRight size={15} />
            </Link>
            <Link href="/clinical" style={{
              display: "inline-flex", alignItems: "center", gap: 8,
              padding: "13px 30px", borderRadius: 12,
              background: "rgba(52,211,153,0.1)", border: "1px solid rgba(52,211,153,0.3)",
              color: "#34d399", fontWeight: 700, fontSize: 14,
              textDecoration: "none", transition: "all 0.2s",
            }}>
              <ClipboardList size={16} /> Clinical Assessment
            </Link>
          </div>
        </div>

        {/* ── Stats ── */}
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit,minmax(210px,1fr))", gap: 16, marginBottom: 64 }}>
          {STATS.map(({ icon: Icon, label, value, color }) => (
            <div key={label}
              className="glass"
              style={{ borderRadius: 18, padding: "22px 24px", display: "flex", alignItems: "center", gap: 18 }}>
              <div style={{
                width: 48, height: 48, borderRadius: 14,
                background: `${color}18`,
                display: "flex", alignItems: "center", justifyContent: "center",
              }}>
                <Icon size={22} color={color} />
              </div>
              <div>
                <div style={{ fontSize: 26, fontWeight: 800, color: "#f1f5f9", letterSpacing: "-0.03em" }}>{value}</div>
                <div style={{ fontSize: 12, color: "#64748b", marginTop: 1 }}>{label}</div>
              </div>
            </div>
          ))}
        </div>

        {/* ── Feature Cards ── */}
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit,minmax(340px,1fr))", gap: 24, marginBottom: 64 }}>
          {FEATURES.map(({ icon: Icon, title, desc, color, href, cta }) => (
            <div key={title}
              className="glass"
              style={{ borderRadius: 22, padding: 32, position: "relative", overflow: "hidden" }}>
              {/* Glow orb */}
              <div style={{
                position: "absolute", top: -40, right: -40, width: 150, height: 150,
                borderRadius: "50%", background: `${color}18`, filter: "blur(40px)",
                pointerEvents: "none",
              }} />

              <div style={{
                width: 52, height: 52, borderRadius: 16,
                background: `${color}18`, border: `1px solid ${color}30`,
                display: "flex", alignItems: "center", justifyContent: "center", marginBottom: 20,
              }}>
                <Icon size={24} color={color} />
              </div>

              <h2 style={{ fontSize: 20, fontWeight: 700, color: "#f1f5f9", marginBottom: 10 }}>{title}</h2>
              <p style={{ fontSize: 14, color: "#64748b", lineHeight: 1.75, marginBottom: 28 }}>{desc}</p>

              <Link href={href} style={{
                display: "inline-flex", alignItems: "center", gap: 8,
                padding: "10px 22px", borderRadius: 10,
                background: `${color}18`, border: `1px solid ${color}30`,
                color, fontWeight: 700, fontSize: 13,
                textDecoration: "none", transition: "all 0.2s",
              }}>
                {cta} <ArrowRight size={14} />
              </Link>
            </div>
          ))}
        </div>

        {/* ── DR Scale Reference ── */}
        <div className="glass" style={{ borderRadius: 22, padding: 36, marginBottom: 64 }}>
          <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 28 }}>
            <BarChart3 size={18} color="#818cf8" />
            <h2 style={{ fontSize: 16, fontWeight: 700, color: "#f1f5f9" }}>
              International Clinical DR Severity Scale
            </h2>
          </div>
          <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
            {DR_LEVELS.map(({ label, grade, color, desc }) => (
              <div key={grade} style={{ display: "flex", alignItems: "center", gap: 16 }}>
                <div style={{
                  width: 32, height: 32, borderRadius: 9,
                  background: `${color}18`, border: `1px solid ${color}30`,
                  display: "flex", alignItems: "center", justifyContent: "center",
                  fontSize: 13, fontWeight: 700, color, flexShrink: 0,
                }}>
                  {grade}
                </div>
                <div style={{ flex: 1 }}>
                  <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 4 }}>
                    <span style={{ fontSize: 14, fontWeight: 600, color }}>{label}</span>
                    <div style={{ flex: 1, height: 4, borderRadius: 100, background: "rgba(255,255,255,0.05)", overflow: "hidden" }}>
                      <div style={{ height: "100%", width: `${20 + grade * 18}%`, background: color, borderRadius: 100 }} />
                    </div>
                    {grade === 0 && <CheckCircle size={13} color={color} />}
                    {grade >= 3 && <AlertTriangle size={13} color={color} />}
                  </div>
                  <span style={{ fontSize: 12, color: "#475569" }}>{desc}</span>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* ── Technical Stack ── */}
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit,minmax(270px,1fr))", gap: 20 }}>
          {[
            {
              icon: Brain, color: "#818cf8", title: "EfficientNetB3 Deep Learning",
              points: ["ImageNet pre-trained weights", "Two-stage transfer learning", "Test-Time Augmentation (TTA)", "Grad-CAM heatmap visualization"]
            },
            {
              icon: BarChart3, color: "#34d399", title: "Random Forest Clinical Model",
              points: ["24 engineered clinical features", "Trained on 300-patient dataset", "Class-weighted for imbalance", "Probability calibration"]
            },
            {
              icon: Shield, color: "#f59e0b", title: "Clinical-Grade Pipeline",
              points: ["CLAHE fundus preprocessing", "Circular crop & normalization", "5-level ICDR severity scale", "Risk factor analysis"]
            },
          ].map(({ icon: Icon, color, title, points }) => (
            <div key={title} className="glass" style={{ borderRadius: 20, padding: "26px 28px" }}>
              <div style={{
                width: 44, height: 44, borderRadius: 14,
                background: `${color}18`, display: "flex", alignItems: "center",
                justifyContent: "center", marginBottom: 18,
              }}>
                <Icon size={22} color={color} />
              </div>
              <h3 style={{ fontSize: 14, fontWeight: 700, color: "#f1f5f9", marginBottom: 14 }}>{title}</h3>
              <ul style={{ listStyle: "none", padding: 0, margin: 0, display: "flex", flexDirection: "column", gap: 8 }}>
                {points.map(pt => (
                  <li key={pt} style={{ display: "flex", alignItems: "center", gap: 8, fontSize: 13, color: "#64748b" }}>
                    <Zap size={11} color={color} />
                    {pt}
                  </li>
                ))}
              </ul>
            </div>
          ))}
        </div>
      </main>

      <footer style={{ borderTop: "1px solid rgba(255,255,255,0.06)", padding: "28px 24px", textAlign: "center" }}>
        <p style={{ fontSize: 12, color: "#334155" }}>
          ⚠️ <strong style={{ color: "#475569" }}>Medical Disclaimer:</strong> For screening assistance and educational purposes only.
          Always consult a licensed ophthalmologist for medical diagnosis and treatment decisions.
        </p>
      </footer>
    </div>
  )
}