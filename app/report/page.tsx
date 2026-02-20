"use client"

import { useState } from "react"
import {
    FileText, Printer, User, Calendar, Pill, ClipboardList,
    AlertTriangle, CheckCircle, ChevronDown, Stethoscope,
    Activity, Eye, Clock, Download
} from "lucide-react"
import { Navbar } from "@/components/Navbar"

// ── Prescription data by DR grade ──────────────────────────────
const PRESCRIPTIONS = [
    {
        grade: 0,
        label: "No Diabetic Retinopathy",
        color: "#22c55e",
        bg: "rgba(34,197,94,0.08)",
        border: "rgba(34,197,94,0.25)",
        diagnosis:
            "No evidence of diabetic retinopathy detected. Retinal vasculature appears structurally intact with no observable microaneurysms, hemorrhages, or exudative changes.",
        medications: [
            { name: "Metformin 500 mg", dose: "1 tablet twice daily with meals", purpose: "Glycemic control" },
            { name: "Aspirin 75 mg", dose: "1 tablet once daily after breakfast", purpose: "Cardiovascular prophylaxis" },
            { name: "Omega-3 Fatty Acids 1000 mg", dose: "1 capsule once daily", purpose: "Lipid management & retinal protection" },
        ],
        eyeDrops: [],
        investigations: [
            "Fasting blood glucose & HbA1c (every 3 months)",
            "Lipid profile (every 6 months)",
            "Urine microalbumin (annually)",
            "Dilated fundus examination (annually)",
            "Blood pressure monitoring (monthly)",
        ],
        followUp: "12 months",
        lifestyle: [
            "Maintain HbA1c < 7.0% through diet and medication",
            "Blood pressure target < 130/80 mmHg",
            "Regular aerobic exercise 30 min/day, 5 days/week",
            "Low glycemic index diet; avoid refined sugars",
            "Smoking cessation strongly advised",
            "Annual comprehensive eye exam",
        ],
        urgency: "Routine",
    },
    {
        grade: 1,
        label: "Mild Non-Proliferative DR",
        color: "#84cc16",
        bg: "rgba(132,204,22,0.08)",
        border: "rgba(132,204,22,0.25)",
        diagnosis:
            "Mild non-proliferative diabetic retinopathy (NPDR) identified. Microaneurysms noted on fundoscopy. No clinically significant macular edema or hard exudates at this stage.",
        medications: [
            { name: "Metformin 1000 mg", dose: "1 tablet twice daily with meals", purpose: "Enhanced glycemic control" },
            { name: "Lisinopril 5 mg", dose: "1 tablet once daily", purpose: "ACE inhibitor – nephro/retinal protection" },
            { name: "Atorvastatin 10 mg", dose: "1 tablet at bedtime", purpose: "Dyslipidemia management" },
            { name: "Vitamin B12 500 mcg", dose: "1 tablet daily", purpose: "Neuropathy prevention" },
        ],
        eyeDrops: [],
        investigations: [
            "HbA1c every 3 months (target < 7.0%)",
            "Optical coherence tomography (OCT) baseline",
            "Dilated fundus exam every 9-12 months",
            "Renal function tests (creatinine, eGFR)",
            "Microalbuminuria screening",
        ],
        followUp: "9 months",
        lifestyle: [
            "Strict glycemic control — HbA1c < 7.0%",
            "Blood pressure strictly < 130/80 mmHg",
            "Dietary sodium restriction < 2,300 mg/day",
            "Avoid smoking — significantly accelerates NPDR progression",
            "Reduce sedentary behavior; include resistance training",
            "Self-monitoring of blood glucose twice daily",
        ],
        urgency: "Monitoring",
    },
    {
        grade: 2,
        label: "Moderate Non-Proliferative DR",
        color: "#f59e0b",
        bg: "rgba(245,158,11,0.08)",
        border: "rgba(245,158,11,0.25)",
        diagnosis:
            "Moderate NPDR identified. Multiple microaneurysms, dot/blot hemorrhages, and hard exudates observed. Risk of macular edema progression warrants close ophthalmologic follow-up.",
        medications: [
            { name: "Metformin 1000 mg + Glipizide 5 mg", dose: "1 tablet twice daily with meals", purpose: "Combined glycemic control" },
            { name: "Ramipril 5 mg", dose: "1 tablet once daily", purpose: "Retinal & renal protection" },
            { name: "Atorvastatin 20 mg", dose: "1 tablet at bedtime", purpose: "Aggressive lipid lowering" },
            { name: "Fenofibrate 145 mg", dose: "1 tablet once daily", purpose: "Triglyceride reduction — retinal benefit" },
            { name: "Multivitamin (Diabetic formula)", dose: "1 tablet daily", purpose: "Antioxidant support" },
        ],
        eyeDrops: [
            { name: "Tropicamide 1% Eye Drops", dose: "2 drops in each eye before fundus exam", purpose: "Pupillary dilation for examination" },
        ],
        investigations: [
            "HbA1c every 3 months",
            "OCT macula — detect subclinical edema",
            "Fluorescein angiography if macular edema suspected",
            "Dilated fundus exam every 6 months",
            "24-hour ambulatory blood pressure monitoring",
            "Kidney function panel every 6 months",
        ],
        followUp: "6 months",
        lifestyle: [
            "Strict HbA1c target < 6.5–7.0%",
            "BP < 125/75 mmHg — stricter target given retinopathy",
            "Carbohydrate counting and portion control",
            "Immediate smoking cessation",
            "Limit alcohol to < 1 unit/day",
            "Adequate sleep (7–8 hours) — impacts insulin sensitivity",
        ],
        urgency: "Referral needed",
    },
    {
        grade: 3,
        label: "Severe Non-Proliferative DR",
        color: "#ef4444",
        bg: "rgba(239,68,68,0.08)",
        border: "rgba(239,68,68,0.25)",
        diagnosis:
            "Severe NPDR identified — 4-2-1 rule criteria met. Extensive intraretinal hemorrhages, venous beading, and IRMA observed. High risk of progression to proliferative DR within 12 months.",
        medications: [
            { name: "Insulin Glargine (Lantus) 10 IU", dose: "Once daily subcutaneous injection at bedtime", purpose: "Basal insulin for tight glycemic control" },
            { name: "Insulin Aspart (NovoRapid)", dose: "Before each meal (dose adjusted by carb count)", purpose: "Prandial glucose control" },
            { name: "Perindopril 4 mg", dose: "1 tablet once daily", purpose: "RAAS blockade — vascular protection" },
            { name: "Amlodipine 5 mg", dose: "1 tablet once daily", purpose: "Blood pressure control" },
            { name: "Rosuvastatin 20 mg", dose: "1 tablet at bedtime", purpose: "High-intensity statin therapy" },
            { name: "Fenofibrate 145 mg", dose: "1 tablet once daily", purpose: "Retinal micro-vascular protection" },
        ],
        eyeDrops: [
            { name: "Lucentis (Ranibizumab 0.5 mg/0.05 mL)", dose: "Intravitreal injection — as scheduled by retinal specialist", purpose: "Anti-VEGF — prevent neovascularization" },
        ],
        investigations: [
            "URGENT: Retinal specialist referral within 4 weeks",
            "Fluorescein angiography (FA)",
            "OCT with angiography (OCT-A)",
            "HbA1c every 3 months (target < 7.0%)",
            "Renal function panel monthly",
            "Complete metabolic panel",
        ],
        followUp: "4–6 weeks (Urgent)",
        lifestyle: [
            "URGENT: Strict blood glucose monitoring 4+ times/day",
            "Avoid heavy lifting or straining — increases IOP risk",
            "Avoid aspirin/NSAIDs unless prescribed",
            "Wear UV-protective sunglasses outdoors",
            "Contact physician immediately if sudden vision changes",
            "Absolute smoking cessation",
        ],
        urgency: "Urgent Referral",
    },
    {
        grade: 4,
        label: "Proliferative Diabetic Retinopathy",
        color: "#dc2626",
        bg: "rgba(220,38,38,0.10)",
        border: "rgba(220,38,38,0.35)",
        diagnosis:
            "Proliferative diabetic retinopathy (PDR) confirmed. Neovascularization of disc (NVD) / elsewhere (NVE) identified. Vitreous hemorrhage risk is high. Immediate ophthalmologic intervention is critical to prevent irreversible vision loss.",
        medications: [
            { name: "Insulin Pump Therapy (CSII)", dose: "Continuous subcutaneous infusion — managed by endocrinologist", purpose: "Intensive glycemic control" },
            { name: "Perindopril/Amlodipine 4/5 mg", dose: "1 tablet once daily", purpose: "Combination BP control" },
            { name: "Rosuvastatin 40 mg", dose: "1 tablet at bedtime", purpose: "Maximum-intensity statin" },
            { name: "Ezetimibe 10 mg", dose: "1 tablet once daily", purpose: "Additional LDL lowering" },
            { name: "Aspirin 75 mg", dose: "1 tablet once daily", purpose: "Antiplatelet — cardiovascular risk" },
        ],
        eyeDrops: [
            { name: "Eylea (Aflibercept 2 mg/0.05 mL)", dose: "Intravitreal injection every 4 weeks × 5, then every 8 weeks", purpose: "Anti-VEGF — neovascularization regression" },
            { name: "Prednisolone Acetate 1% Eye Drops", dose: "1 drop 4 times daily post-procedure", purpose: "Post-injection inflammation control" },
        ],
        investigations: [
            "EMERGENCY: Immediate retinal specialist referral",
            "B-scan ultrasonography (if vitreous hemorrhage present)",
            "Fluorescein angiography",
            "OCT-Angiography",
            "Pan-retinal photocoagulation (PRP) laser evaluation",
            "Vitrectomy candidacy assessment",
        ],
        followUp: "1–2 weeks (Emergency)",
        lifestyle: [
            "EMERGENCY: Immediate ophthalmology appointment",
            "Bed rest + avoid strenuous activity",
            "No reading or screen use if vision acuity drastically reduced",
            "Alert ER if sudden floaters, flashes, or curtain-like vision loss",
            "Complete smoking cessation — most critical step",
            "Emotional support/counselling — psychological impact of vision threat",
        ],
        urgency: "EMERGENCY",
    },
]

const GRADE_OPTIONS = PRESCRIPTIONS.map((p) => `Grade ${p.grade} — ${p.label}`)

const inputSt = {
    width: "100%", padding: "10px 14px", borderRadius: 10,
    fontSize: 13, background: "rgba(255,255,255,0.05)",
    border: "1px solid rgba(255,255,255,0.1)", color: "#f1f5f9",
    outline: "none", boxSizing: "border-box" as const,
}

const labelSt = {
    display: "block" as const, fontSize: 11, fontWeight: 700,
    color: "#64748b", marginBottom: 6, letterSpacing: "0.04em",
}

function Fld({ label, children }: { label: string; children: React.ReactNode }) {
    return <div><label style={labelSt}>{label}</label>{children}</div>
}

function Sel({ value, onChange, opts }: { value: string; onChange: (v: string) => void; opts: string[] }) {
    return (
        <div style={{ position: "relative" }}>
            <select value={value} onChange={e => onChange(e.target.value)}
                style={{ ...inputSt, appearance: "none", paddingRight: 32, cursor: "pointer" }}>
                {opts.map(o => <option key={o} value={o} style={{ background: "#0d1530" }}>{o}</option>)}
            </select>
            <ChevronDown size={14} color="#64748b"
                style={{ position: "absolute", right: 10, top: "50%", transform: "translateY(-50%)", pointerEvents: "none" }} />
        </div>
    )
}

export default function ReportPage() {
    const today = new Date().toISOString().split("T")[0]

    const [patientName, setPatientName] = useState("")
    const [patientAge, setPatientAge] = useState("")
    const [patientGender, setPatientGender] = useState("Male")
    const [patientId, setPatientId] = useState("")
    const [doctorName, setDoctorName] = useState("")
    const [hospital, setHospital] = useState("")
    const [selectedGrade, setSelectedGrade] = useState(GRADE_OPTIONS[0])
    const [generated, setGenerated] = useState(false)

    const gradeIndex = parseInt(selectedGrade.split(" ")[1])
    const rx = PRESCRIPTIONS[gradeIndex]

    const generate = () => {
        if (!patientName.trim()) { alert("Please enter patient name."); return }
        setGenerated(true)
        setTimeout(() => {
            document.getElementById("rx-output")?.scrollIntoView({ behavior: "smooth" })
        }, 100)
    }

    const printRx = () => window.print()

    return (
        <div style={{ minHeight: "100vh", color: "#e2e8f0", fontFamily: "'Inter', sans-serif" }}>
            <Navbar />

            {/* Print-only stylesheet */}
            <style>{`
        @media print {
          body { background: white !important; color: black !important; }
          .no-print { display: none !important; }
          #rx-output { box-shadow: none !important; border: 1px solid #ddd !important; }
        }
      `}</style>

            <main style={{ maxWidth: 1100, margin: "0 auto", padding: "48px 24px 80px" }}>

                {/* ── Page header ── */}
                <div style={{ marginBottom: 40 }} className="no-print">
                    <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 10 }}>
                        <div style={{
                            width: 36, height: 36, borderRadius: 10,
                            background: "rgba(129,140,248,0.15)", border: "1px solid rgba(129,140,248,0.25)",
                            display: "flex", alignItems: "center", justifyContent: "center",
                        }}>
                            <FileText size={18} color="#818cf8" />
                        </div>
                        <h1 style={{ fontSize: 24, fontWeight: 800, color: "#f1f5f9", letterSpacing: "-0.02em" }}>
                            Prescription Generator
                        </h1>
                    </div>
                    <p style={{ fontSize: 14, color: "#475569", maxWidth: 620, lineHeight: 1.7, marginLeft: 46 }}>
                        Enter patient details and select the DR grade. A complete clinical prescription with medications,
                        investigations, follow-up schedule, and lifestyle advice will be generated.
                    </p>
                </div>

                <div style={{ display: "grid", gridTemplateColumns: generated ? "1fr 1fr" : "1fr", gap: 28, alignItems: "start" }}>

                    {/* ── Form ── */}
                    <div className="glass no-print" style={{ borderRadius: 22, padding: 32 }}>
                        <h2 style={{ fontSize: 13, fontWeight: 700, color: "#64748b", marginBottom: 28, letterSpacing: "0.05em" }}>
                            PATIENT & DIAGNOSIS DETAILS
                        </h2>

                        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 18, marginBottom: 18 }}>
                            <Fld label="PATIENT FULL NAME">
                                <input value={patientName} onChange={e => setPatientName(e.target.value)}
                                    placeholder="e.g. Rajesh Kumar" style={inputSt} />
                            </Fld>
                            <Fld label="PATIENT ID / MRN">
                                <input value={patientId} onChange={e => setPatientId(e.target.value)}
                                    placeholder="e.g. MRN-20240001" style={inputSt} />
                            </Fld>
                            <Fld label="AGE">
                                <input type="number" value={patientAge} onChange={e => setPatientAge(e.target.value)}
                                    placeholder="e.g. 54" min="1" max="120" style={inputSt} />
                            </Fld>
                            <Fld label="GENDER">
                                <Sel value={patientGender} onChange={setPatientGender} opts={["Male", "Female", "Other"]} />
                            </Fld>
                            <Fld label="CONSULTING DOCTOR">
                                <input value={doctorName} onChange={e => setDoctorName(e.target.value)}
                                    placeholder="Dr. Suresh Reddy" style={inputSt} />
                            </Fld>
                            <Fld label="HOSPITAL / CLINIC">
                                <input value={hospital} onChange={e => setHospital(e.target.value)}
                                    placeholder="City Eye & Diabetes Clinic" style={inputSt} />
                            </Fld>
                        </div>

                        <Fld label="DR GRADE (FROM AI ANALYSIS)">
                            <Sel value={selectedGrade} onChange={val => { setSelectedGrade(val); setGenerated(false) }}
                                opts={GRADE_OPTIONS} />
                        </Fld>

                        {/* Grade preview badge */}
                        <div style={{
                            marginTop: 16, padding: "12px 16px", background: rx.bg,
                            border: `1px solid ${rx.border}`, borderRadius: 12,
                            display: "flex", alignItems: "center", justifyContent: "space-between"
                        }}>
                            <div style={{ fontSize: 13, fontWeight: 700, color: rx.color }}>{rx.label}</div>
                            <div style={{
                                fontSize: 11, fontWeight: 700, color: rx.color,
                                padding: "3px 10px", background: `${rx.color}18`, borderRadius: 100,
                                border: `1px solid ${rx.color}30`
                            }}>
                                {rx.urgency}
                            </div>
                        </div>

                        <button onClick={generate}
                            style={{
                                marginTop: 28, width: "100%", padding: 14, borderRadius: 12,
                                background: "linear-gradient(135deg, #818cf8, #6366f1)",
                                border: "none", color: "#fff", fontWeight: 700, fontSize: 14,
                                cursor: "pointer", display: "flex", alignItems: "center",
                                justifyContent: "center", gap: 10,
                                boxShadow: "0 0 24px rgba(99,102,241,0.35)",
                            }}>
                            <FileText size={16} /> Generate Prescription
                        </button>
                    </div>

                    {/* ── Prescription Output ── */}
                    {generated && (
                        <div id="rx-output" style={{
                            background: "#fff", borderRadius: 20, overflow: "hidden",
                            boxShadow: "0 8px 60px rgba(0,0,0,0.4)",
                            color: "#1e293b", fontFamily: "'Inter', sans-serif",
                        }}>

                            {/* RX Header */}
                            <div style={{
                                background: "linear-gradient(135deg, #1e3a5f 0%, #1e293b 100%)",
                                padding: "28px 32px", color: "#fff",
                            }}>
                                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start" }}>
                                    <div>
                                        <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 8 }}>
                                            <Stethoscope size={22} color="#a78bfa" />
                                            <span style={{ fontSize: 22, fontWeight: 900, letterSpacing: "-0.02em" }}>
                                                {hospital || "RetinaAI Ophthalmology Clinic"}
                                            </span>
                                        </div>
                                        <div style={{ fontSize: 13, color: "#94a3b8" }}>Diabetic Retinopathy & Diabetes Management</div>
                                        <div style={{ fontSize: 12, color: "#64748b", marginTop: 3 }}>
                                            Dr. {doctorName || "Ophthalmologist"} · MBBS, MS Ophthalmology
                                        </div>
                                    </div>
                                    <div style={{ textAlign: "right" }}>
                                        <div style={{ fontSize: 11, color: "#64748b" }}>Date</div>
                                        <div style={{ fontSize: 14, fontWeight: 700, color: "#e2e8f0" }}>
                                            {new Date().toLocaleDateString("en-IN", { day: "2-digit", month: "long", year: "numeric" })}
                                        </div>
                                        <div style={{ fontSize: 11, color: "#64748b", marginTop: 4 }}>Rx No.</div>
                                        <div style={{ fontSize: 12, fontWeight: 700, color: "#818cf8" }}>
                                            RX-{Date.now().toString().slice(-6)}
                                        </div>
                                    </div>
                                </div>
                            </div>

                            <div style={{ padding: "24px 32px" }}>

                                {/* Patient Info */}
                                <div style={{
                                    display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 14, marginBottom: 22,
                                    background: "#f8fafc", borderRadius: 12, padding: "16px 18px", border: "1px solid #e2e8f0"
                                }}>
                                    <div>
                                        <div style={{ fontSize: 10, color: "#94a3b8", fontWeight: 700, letterSpacing: "0.05em" }}>PATIENT</div>
                                        <div style={{ fontSize: 15, fontWeight: 800, color: "#0f172a", marginTop: 2 }}>{patientName || "—"}</div>
                                    </div>
                                    <div>
                                        <div style={{ fontSize: 10, color: "#94a3b8", fontWeight: 700, letterSpacing: "0.05em" }}>AGE / GENDER</div>
                                        <div style={{ fontSize: 15, fontWeight: 800, color: "#0f172a", marginTop: 2 }}>
                                            {patientAge || "—"} yrs / {patientGender}
                                        </div>
                                    </div>
                                    <div>
                                        <div style={{ fontSize: 10, color: "#94a3b8", fontWeight: 700, letterSpacing: "0.05em" }}>ID / MRN</div>
                                        <div style={{ fontSize: 15, fontWeight: 800, color: "#0f172a", marginTop: 2 }}>{patientId || "—"}</div>
                                    </div>
                                </div>

                                {/* Diagnosis */}
                                <div style={{
                                    marginBottom: 20, padding: "16px 18px",
                                    background: rx.bg, border: `2px solid ${rx.border}`, borderRadius: 12
                                }}>
                                    <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 8 }}>
                                        <Eye size={15} color={rx.color} />
                                        <span style={{ fontSize: 11, fontWeight: 800, color: rx.color, letterSpacing: "0.06em" }}>
                                            DIAGNOSIS — {rx.label.toUpperCase()}
                                        </span>
                                        <span style={{
                                            marginLeft: "auto", fontSize: 11, fontWeight: 700, color: rx.color,
                                            padding: "2px 10px", background: `${rx.color}18`,
                                            borderRadius: 100, border: `1px solid ${rx.color}30`
                                        }}>
                                            ⚡ {rx.urgency}
                                        </span>
                                    </div>
                                    <p style={{ fontSize: 12, color: "#374151", lineHeight: 1.75, margin: 0 }}>{rx.diagnosis}</p>
                                </div>

                                {/* Medications */}
                                <div style={{ marginBottom: 18 }}>
                                    <div style={{ display: "flex", alignItems: "center", gap: 7, marginBottom: 12 }}>
                                        <Pill size={15} color="#6366f1" />
                                        <span style={{ fontSize: 12, fontWeight: 800, color: "#1e293b", letterSpacing: "0.04em" }}>
                                            ℞ MEDICATIONS
                                        </span>
                                    </div>
                                    <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
                                        {rx.medications.map((m, i) => (
                                            <div key={i} style={{
                                                display: "grid", gridTemplateColumns: "24px 1fr",
                                                gap: 12, padding: "12px 14px", background: "#f8fafc",
                                                borderRadius: 10, border: "1px solid #e2e8f0"
                                            }}>
                                                <div style={{ fontSize: 14, fontWeight: 800, color: "#6366f1", alignSelf: "flex-start", paddingTop: 1 }}>
                                                    {i + 1}.
                                                </div>
                                                <div>
                                                    <div style={{ fontSize: 13, fontWeight: 800, color: "#0f172a" }}>{m.name}</div>
                                                    <div style={{ fontSize: 12, color: "#475569", marginTop: 2 }}>
                                                        <strong>Sig:</strong> {m.dose}
                                                    </div>
                                                    <div style={{ fontSize: 11, color: "#94a3b8", marginTop: 1 }}>
                                                        Purpose: {m.purpose}
                                                    </div>
                                                </div>
                                            </div>
                                        ))}
                                        {rx.eyeDrops.map((m, i) => (
                                            <div key={`eye-${i}`} style={{
                                                display: "grid", gridTemplateColumns: "24px 1fr",
                                                gap: 12, padding: "12px 14px", background: "#fff7ed",
                                                borderRadius: 10, border: "1px solid #fed7aa"
                                            }}>
                                                <div style={{ fontSize: 14, fontWeight: 800, color: "#ea580c", alignSelf: "flex-start", paddingTop: 1 }}>
                                                    👁
                                                </div>
                                                <div>
                                                    <div style={{ fontSize: 13, fontWeight: 800, color: "#0f172a" }}>{m.name}</div>
                                                    <div style={{ fontSize: 12, color: "#475569", marginTop: 2 }}>
                                                        <strong>Sig:</strong> {m.dose}
                                                    </div>
                                                    <div style={{ fontSize: 11, color: "#94a3b8", marginTop: 1 }}>
                                                        Purpose: {m.purpose}
                                                    </div>
                                                </div>
                                            </div>
                                        ))}
                                    </div>
                                </div>

                                {/* Investigations */}
                                <div style={{ marginBottom: 18 }}>
                                    <div style={{ display: "flex", alignItems: "center", gap: 7, marginBottom: 12 }}>
                                        <Activity size={15} color="#10b981" />
                                        <span style={{ fontSize: 12, fontWeight: 800, color: "#1e293b", letterSpacing: "0.04em" }}>
                                            INVESTIGATIONS ADVISED
                                        </span>
                                    </div>
                                    <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
                                        {rx.investigations.map((inv, i) => (
                                            <div key={i} style={{
                                                display: "flex", alignItems: "center", gap: 10,
                                                padding: "8px 12px", background: "#f0fdf4", borderRadius: 8, border: "1px solid #bbf7d0"
                                            }}>
                                                <CheckCircle size={12} color="#10b981" style={{ flexShrink: 0 }} />
                                                <span style={{ fontSize: 12, color: "#065f46" }}>{inv}</span>
                                            </div>
                                        ))}
                                    </div>
                                </div>

                                {/* Follow-up + Lifestyle in 2 cols */}
                                <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16, marginBottom: 20 }}>
                                    {/* Follow-up */}
                                    <div style={{
                                        padding: "14px 16px",
                                        background: gradeIndex >= 3 ? "#fff1f2" : "#eff6ff",
                                        borderRadius: 12, border: gradeIndex >= 3 ? "1px solid #fecdd3" : "1px solid #bfdbfe"
                                    }}>
                                        <div style={{ display: "flex", alignItems: "center", gap: 7, marginBottom: 10 }}>
                                            <Clock size={13} color={gradeIndex >= 3 ? "#ef4444" : "#2563eb"} />
                                            <span style={{
                                                fontSize: 11, fontWeight: 800, color: gradeIndex >= 3 ? "#ef4444" : "#2563eb",
                                                letterSpacing: "0.05em"
                                            }}>NEXT FOLLOW-UP</span>
                                        </div>
                                        <div style={{ fontSize: 18, fontWeight: 900, color: gradeIndex >= 3 ? "#dc2626" : "#1d4ed8" }}>
                                            {rx.followUp}
                                        </div>
                                        {gradeIndex >= 3 && (
                                            <div style={{
                                                display: "flex", alignItems: "center", gap: 5, marginTop: 6,
                                                fontSize: 11, color: "#ef4444", fontWeight: 600
                                            }}>
                                                <AlertTriangle size={11} /> Do not delay — vision at risk
                                            </div>
                                        )}
                                    </div>

                                    {/* Calendar note */}
                                    <div style={{
                                        padding: "14px 16px", background: "#faf5ff",
                                        borderRadius: 12, border: "1px solid #e9d5ff"
                                    }}>
                                        <div style={{ display: "flex", alignItems: "center", gap: 7, marginBottom: 10 }}>
                                            <Calendar size={13} color="#7c3aed" />
                                            <span style={{ fontSize: 11, fontWeight: 800, color: "#7c3aed", letterSpacing: "0.05em" }}>
                                                APPOINTMENT DATE
                                            </span>
                                        </div>
                                        <div style={{ fontSize: 18, fontWeight: 900, color: "#6d28d9" }}>
                                            {(() => {
                                                const d = new Date()
                                                const months = [0, 9, 6, 3, 1, 1]
                                                d.setMonth(d.getMonth() + (months[gradeIndex] || 1))
                                                return d.toLocaleDateString("en-IN", { day: "2-digit", month: "long", year: "numeric" })
                                            })()}
                                        </div>
                                        <div style={{ fontSize: 11, color: "#64748b", marginTop: 2 }}>Estimated next review</div>
                                    </div>
                                </div>

                                {/* Lifestyle */}
                                <div style={{
                                    marginBottom: 22, padding: "16px 18px",
                                    background: "#f8fafc", borderRadius: 12, border: "1px solid #e2e8f0"
                                }}>
                                    <div style={{ display: "flex", alignItems: "center", gap: 7, marginBottom: 12 }}>
                                        <ClipboardList size={14} color="#0ea5e9" />
                                        <span style={{ fontSize: 11, fontWeight: 800, color: "#0ea5e9", letterSpacing: "0.05em" }}>
                                            LIFESTYLE & DIETARY ADVICE
                                        </span>
                                    </div>
                                    <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "6px 16px" }}>
                                        {rx.lifestyle.map((l, i) => (
                                            <div key={i} style={{ display: "flex", alignItems: "flex-start", gap: 7, fontSize: 11, color: "#374151", lineHeight: 1.6 }}>
                                                <span style={{ color: "#0ea5e9", marginTop: 2, flexShrink: 0 }}>→</span> {l}
                                            </div>
                                        ))}
                                    </div>
                                </div>

                                {/* Footer / Signature */}
                                <div style={{
                                    borderTop: "2px dashed #e2e8f0", paddingTop: 18,
                                    display: "flex", justifyContent: "space-between", alignItems: "flex-end"
                                }}>
                                    <div style={{ fontSize: 10, color: "#94a3b8", maxWidth: 340, lineHeight: 1.6 }}>
                                        ⚠️ <strong>Disclaimer:</strong> This prescription is generated by the RetinaAI screening tool as a clinical aid only.
                                        It does <em>not</em> replace consultation with a licensed ophthalmologist or physician. Dosages must be verified
                                        and adjusted by a qualified medical professional.
                                    </div>
                                    <div style={{ textAlign: "center" }}>
                                        <div style={{ width: 150, height: 1, background: "#64748b", marginBottom: 6 }} />
                                        <div style={{ fontSize: 12, fontWeight: 700, color: "#0f172a" }}>
                                            Dr. {doctorName || "__________________"}
                                        </div>
                                        <div style={{ fontSize: 10, color: "#94a3b8" }}>Signature & Stamp</div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    )}
                </div>

                {/* Print/Download buttons */}
                {generated && (
                    <div className="no-print" style={{ display: "flex", gap: 14, marginTop: 24, justifyContent: "flex-end" }}>
                        <button onClick={printRx}
                            style={{
                                display: "flex", alignItems: "center", gap: 8,
                                padding: "11px 24px", borderRadius: 11,
                                background: "rgba(129,140,248,0.12)", border: "1px solid rgba(129,140,248,0.3)",
                                color: "#818cf8", fontWeight: 700, fontSize: 13, cursor: "pointer",
                            }}>
                            <Printer size={15} /> Print Prescription
                        </button>
                        <button onClick={printRx}
                            style={{
                                display: "flex", alignItems: "center", gap: 8,
                                padding: "11px 24px", borderRadius: 11,
                                background: "linear-gradient(135deg, #818cf8, #6366f1)",
                                border: "none", color: "#fff", fontWeight: 700, fontSize: 13, cursor: "pointer",
                                boxShadow: "0 0 20px rgba(99,102,241,0.3)",
                            }}>
                            <Download size={15} /> Save as PDF
                        </button>
                    </div>
                )}
            </main>

            <footer style={{ borderTop: "1px solid rgba(255,255,255,0.06)", padding: "24px", textAlign: "center" }}>
                <p style={{ fontSize: 12, color: "#334155" }}>
                    ⚠️ Prescriptions generated are for reference only. Always consult a licensed ophthalmologist.
                </p>
            </footer>
        </div>
    )
}
