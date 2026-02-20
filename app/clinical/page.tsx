"use client"

import { useState } from "react"
import {
    ClipboardList, Activity, AlertTriangle, CheckCircle,
    Info, ChevronDown, BarChart3, Brain, Loader2
} from "lucide-react"
import { Navbar } from "@/components/Navbar"

const API_BASE = "http://localhost:8000"

const SEVERITY = [
    { label: "No DR", color: "#22c55e", bg: "rgba(34,197,94,0.10)", border: "rgba(34,197,94,0.25)" },
    { label: "Mild DR", color: "#84cc16", bg: "rgba(132,204,22,0.10)", border: "rgba(132,204,22,0.25)" },
    { label: "Moderate DR", color: "#f59e0b", bg: "rgba(245,158,11,0.10)", border: "rgba(245,158,11,0.25)" },
    { label: "Severe DR", color: "#ef4444", bg: "rgba(239,68,68,0.10)", border: "rgba(239,68,68,0.25)" },
    { label: "Proliferative DR", color: "#dc2626", bg: "rgba(220,38,38,0.14)", border: "rgba(220,38,38,0.35)" },
]

const VA_OPTIONS = ["20/20", "20/25", "20/30", "20/40", "20/50", "20/60", "20/80", "20/100", "20/200", "CF", "HM", "LP", "NLP"]

type ClinicalResult = {
    grade: number
    class_name: string
    confidence: number
    probabilities: Record<string, number>
    risk_level: string
    recommendation: string
    risk_factors: string[]
    processing_time: number
    model: string
    demo_mode: boolean
}

type FormData = {
    age: string; diabetes_duration: string; hba1c: string
    systolic_bp: string; diastolic_bp: string; bmi: string
    cholesterol: string; iop: string; va_right: string; va_left: string
    gender: string; ethnicity: string; diabetes_type: string
    smoking: string; kidney_disease: string; neuropathy: string
    eye_side: string; has_symptom: boolean
}

const INITIAL: FormData = {
    age: "", diabetes_duration: "", hba1c: "", systolic_bp: "", diastolic_bp: "",
    bmi: "", cholesterol: "", iop: "", va_right: "20/20", va_left: "20/20",
    gender: "Male", ethnicity: "Asian", diabetes_type: "Type 2",
    smoking: "Never", kidney_disease: "No", neuropathy: "No",
    eye_side: "Both", has_symptom: false,
}

const inputStyle = {
    width: "100%", padding: "10px 14px", borderRadius: 10, fontSize: 13,
    background: "rgba(255,255,255,0.04)", border: "1px solid rgba(255,255,255,0.1)",
    color: "#f1f5f9", outline: "none", boxSizing: "border-box" as const,
}

const labelStyle = {
    display: "block", fontSize: 11, fontWeight: 700, color: "#64748b",
    marginBottom: 6, letterSpacing: "0.04em",
}

function Field({ label, children }: { label: string; children: React.ReactNode }) {
    return (
        <div>
            <label style={labelStyle}>{label}</label>
            {children}
        </div>
    )
}

function NumInput({ value, onChange, placeholder, min, max, step = "0.1" }: {
    value: string; onChange: (v: string) => void; placeholder: string;
    min?: string; max?: string; step?: string
}) {
    return (
        <input type="number" value={value} onChange={e => onChange(e.target.value)}
            placeholder={placeholder} min={min} max={max} step={step}
            style={inputStyle} />
    )
}

function SelectInput({ value, onChange, options }: {
    value: string; onChange: (v: string) => void; options: string[]
}) {
    return (
        <div style={{ position: "relative" }}>
            <select value={value} onChange={e => onChange(e.target.value)}
                style={{ ...inputStyle, appearance: "none", paddingRight: 32, cursor: "pointer" }}>
                {options.map(o => <option key={o} value={o} style={{ background: "#0d1530" }}>{o}</option>)}
            </select>
            <ChevronDown size={14} color="#64748b" style={{ position: "absolute", right: 10, top: "50%", transform: "translateY(-50%)", pointerEvents: "none" }} />
        </div>
    )
}

export default function ClinicalPage() {
    const [form, setForm] = useState<FormData>(INITIAL)
    const [result, setResult] = useState<ClinicalResult | null>(null)
    const [loading, setLoading] = useState(false)
    const [error, setError] = useState<string | null>(null)

    const set = (k: keyof FormData) => (v: string | boolean) =>
        setForm(prev => ({ ...prev, [k]: v }))

    const handleSubmit = async () => {
        // Validate required numerics
        const reqFields: (keyof FormData)[] = [
            "age", "diabetes_duration", "hba1c", "systolic_bp", "diastolic_bp", "bmi", "cholesterol", "iop"
        ]
        for (const f of reqFields) {
            if (!form[f]) {
                setError(`Please fill in the "${f.replace(/_/g, " ")}" field.`)
                return
            }
        }

        setLoading(true); setError(null); setResult(null)
        try {
            const body = {
                age: parseFloat(form.age),
                diabetes_duration: parseFloat(form.diabetes_duration),
                hba1c: parseFloat(form.hba1c),
                systolic_bp: parseFloat(form.systolic_bp),
                diastolic_bp: parseFloat(form.diastolic_bp),
                bmi: parseFloat(form.bmi),
                cholesterol: parseFloat(form.cholesterol),
                iop: parseFloat(form.iop),
                va_right: form.va_right,
                va_left: form.va_left,
                gender: form.gender,
                ethnicity: form.ethnicity,
                diabetes_type: form.diabetes_type,
                smoking: form.smoking,
                kidney_disease: form.kidney_disease,
                neuropathy: form.neuropathy,
                eye_side: form.eye_side,
                has_symptom: form.has_symptom,
            }

            const res = await fetch(`${API_BASE}/predict-clinical`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(body),
                signal: AbortSignal.timeout(30000),
            })

            if (!res.ok) {
                const err = await res.json().catch(() => ({ detail: "Unknown error" }))
                throw new Error(err.detail || `HTTP ${res.status}`)
            }
            const data: ClinicalResult = await res.json()
            setResult(data)
        } catch (e: any) {
            if (e.name === "TimeoutError" || e.name === "TypeError") {
                setError("Cannot reach the backend API. Make sure the FastAPI server is running on port 8000.")
            } else {
                setError(e.message || "Clinical assessment failed. Please try again.")
            }
        } finally {
            setLoading(false)
        }
    }

    const level = result !== null ? SEVERITY[result.grade] : null
    const probs = result ? Object.entries(result.probabilities) : []

    return (
        <div style={{ minHeight: "100vh", color: "#e2e8f0", fontFamily: "'Inter', sans-serif" }}>
            <Navbar />

            <main style={{ maxWidth: 1200, margin: "0 auto", padding: "48px 24px 80px" }}>
                {/* Header */}
                <div style={{ marginBottom: 40 }}>
                    <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 10 }}>
                        <div style={{
                            width: 36, height: 36, borderRadius: 10,
                            background: "rgba(52,211,153,0.12)", border: "1px solid rgba(52,211,153,0.25)",
                            display: "flex", alignItems: "center", justifyContent: "center",
                        }}>
                            <ClipboardList size={18} color="#34d399" />
                        </div>
                        <h1 style={{ fontSize: 24, fontWeight: 800, color: "#f1f5f9", letterSpacing: "-0.02em" }}>
                            Clinical Assessment
                        </h1>
                    </div>
                    <p style={{ fontSize: 14, color: "#475569", maxWidth: 600, lineHeight: 1.7, marginLeft: 46 }}>
                        Enter patient clinical and demographic data. The Random Forest model trained on 300 diabetic patients
                        will predict DR severity using 24 engineered features.
                    </p>
                </div>

                <div style={{ display: "grid", gridTemplateColumns: result ? "1fr 420px" : "1fr", gap: 28, alignItems: "start" }}>

                    {/* ── Form ── */}
                    <div className="glass" style={{ borderRadius: 22, padding: 32 }}>
                        <h2 style={{ fontSize: 13, fontWeight: 700, color: "#64748b", marginBottom: 28, letterSpacing: "0.05em" }}>
                            PATIENT CLINICAL DATA
                        </h2>

                        {/* Demographics */}
                        <div style={{ marginBottom: 28 }}>
                            <div style={{ fontSize: 12, fontWeight: 700, color: "#818cf8", marginBottom: 16, display: "flex", alignItems: "center", gap: 6 }}>
                                <div style={{ width: 4, height: 14, borderRadius: 2, background: "#818cf8" }} /> Demographics
                            </div>
                            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit,minmax(160px,1fr))", gap: 16 }}>
                                <Field label="AGE (years)">
                                    <NumInput value={form.age} onChange={set("age")} placeholder="e.g. 55" min="0" max="120" step="1" />
                                </Field>
                                <Field label="GENDER">
                                    <SelectInput value={form.gender} onChange={set("gender") as any} options={["Male", "Female", "Other"]} />
                                </Field>
                                <Field label="ETHNICITY">
                                    <SelectInput value={form.ethnicity} onChange={set("ethnicity") as any} options={["Asian", "Black", "Hispanic", "White", "Other"]} />
                                </Field>
                            </div>
                        </div>

                        {/* Diabetes info */}
                        <div style={{ marginBottom: 28 }}>
                            <div style={{ fontSize: 12, fontWeight: 700, color: "#34d399", marginBottom: 16, display: "flex", alignItems: "center", gap: 6 }}>
                                <div style={{ width: 4, height: 14, borderRadius: 2, background: "#34d399" }} /> Diabetes Information
                            </div>
                            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit,minmax(160px,1fr))", gap: 16 }}>
                                <Field label="DIABETES TYPE">
                                    <SelectInput value={form.diabetes_type} onChange={set("diabetes_type") as any} options={["Type 1", "Type 2"]} />
                                </Field>
                                <Field label="DURATION (years)">
                                    <NumInput value={form.diabetes_duration} onChange={set("diabetes_duration")} placeholder="e.g. 8" min="0" max="80" step="0.5" />
                                </Field>
                                <Field label="HbA1c (%)">
                                    <NumInput value={form.hba1c} onChange={set("hba1c")} placeholder="e.g. 7.5" min="3" max="20" />
                                </Field>
                            </div>
                        </div>

                        {/* Vitals */}
                        <div style={{ marginBottom: 28 }}>
                            <div style={{ fontSize: 12, fontWeight: 700, color: "#f59e0b", marginBottom: 16, display: "flex", alignItems: "center", gap: 6 }}>
                                <div style={{ width: 4, height: 14, borderRadius: 2, background: "#f59e0b" }} /> Vital Signs & Lab Values
                            </div>
                            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit,minmax(150px,1fr))", gap: 16 }}>
                                <Field label="SYSTOLIC BP (mmHg)">
                                    <NumInput value={form.systolic_bp} onChange={set("systolic_bp")} placeholder="e.g. 130" min="60" max="300" step="1" />
                                </Field>
                                <Field label="DIASTOLIC BP (mmHg)">
                                    <NumInput value={form.diastolic_bp} onChange={set("diastolic_bp")} placeholder="e.g. 85" min="40" max="200" step="1" />
                                </Field>
                                <Field label="BMI">
                                    <NumInput value={form.bmi} onChange={set("bmi")} placeholder="e.g. 27.4" min="10" max="70" />
                                </Field>
                                <Field label="CHOLESTEROL (mg/dL)">
                                    <NumInput value={form.cholesterol} onChange={set("cholesterol")} placeholder="e.g. 195" min="50" max="500" step="1" />
                                </Field>
                            </div>
                        </div>

                        {/* Ophthalmic */}
                        <div style={{ marginBottom: 28 }}>
                            <div style={{ fontSize: 12, fontWeight: 700, color: "#f87171", marginBottom: 16, display: "flex", alignItems: "center", gap: 6 }}>
                                <div style={{ width: 4, height: 14, borderRadius: 2, background: "#f87171" }} /> Ophthalmological Assessment
                            </div>
                            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit,minmax(150px,1fr))", gap: 16 }}>
                                <Field label="IOP (mmHg)">
                                    <NumInput value={form.iop} onChange={set("iop")} placeholder="e.g. 16" min="5" max="50" step="0.5" />
                                </Field>
                                <Field label="VA RIGHT EYE">
                                    <SelectInput value={form.va_right} onChange={set("va_right") as any} options={VA_OPTIONS} />
                                </Field>
                                <Field label="VA LEFT EYE">
                                    <SelectInput value={form.va_left} onChange={set("va_left") as any} options={VA_OPTIONS} />
                                </Field>
                                <Field label="EYE SIDE AFFECTED">
                                    <SelectInput value={form.eye_side} onChange={set("eye_side") as any} options={["Right", "Left", "Both"]} />
                                </Field>
                            </div>
                        </div>

                        {/* Comorbidities */}
                        <div style={{ marginBottom: 32 }}>
                            <div style={{ fontSize: 12, fontWeight: 700, color: "#94a3b8", marginBottom: 16, display: "flex", alignItems: "center", gap: 6 }}>
                                <div style={{ width: 4, height: 14, borderRadius: 2, background: "#94a3b8" }} /> Comorbidities & Risk Factors
                            </div>
                            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit,minmax(150px,1fr))", gap: 16 }}>
                                <Field label="SMOKING STATUS">
                                    <SelectInput value={form.smoking} onChange={set("smoking") as any} options={["Never", "Former", "Current"]} />
                                </Field>
                                <Field label="KIDNEY DISEASE">
                                    <SelectInput value={form.kidney_disease} onChange={set("kidney_disease") as any} options={["No", "Yes"]} />
                                </Field>
                                <Field label="NEUROPATHY">
                                    <SelectInput value={form.neuropathy} onChange={set("neuropathy") as any} options={["No", "Yes"]} />
                                </Field>
                                <Field label="VISUAL SYMPTOMS">
                                    <div
                                        onClick={() => set("has_symptom")(!form.has_symptom)}
                                        style={{
                                            display: "flex", alignItems: "center", gap: 10,
                                            padding: "10px 14px", borderRadius: 10, cursor: "pointer",
                                            background: form.has_symptom ? "rgba(239,68,68,0.1)" : "rgba(255,255,255,0.04)",
                                            border: form.has_symptom ? "1px solid rgba(239,68,68,0.3)" : "1px solid rgba(255,255,255,0.1)",
                                            transition: "all 0.2s",
                                        }}>
                                        <div style={{
                                            width: 16, height: 16, borderRadius: 4,
                                            background: form.has_symptom ? "#ef4444" : "rgba(255,255,255,0.08)",
                                            border: form.has_symptom ? "none" : "1px solid rgba(255,255,255,0.2)",
                                            display: "flex", alignItems: "center", justifyContent: "center",
                                        }}>
                                            {form.has_symptom && <CheckCircle size={11} color="#fff" />}
                                        </div>
                                        <span style={{ fontSize: 13, color: form.has_symptom ? "#f87171" : "#64748b" }}>
                                            {form.has_symptom ? "Symptoms present" : "No symptoms"}
                                        </span>
                                    </div>
                                </Field>
                            </div>
                        </div>

                        {/* Error */}
                        {error && (
                            <div style={{
                                marginBottom: 20, padding: "12px 16px",
                                background: "rgba(239,68,68,0.09)", border: "1px solid rgba(239,68,68,0.25)",
                                borderRadius: 10, display: "flex", gap: 10, alignItems: "center"
                            }}>
                                <AlertTriangle size={15} color="#f87171" />
                                <span style={{ fontSize: 13, color: "#f87171" }}>{error}</span>
                            </div>
                        )}

                        {/* Submit */}
                        <button
                            onClick={handleSubmit}
                            disabled={loading}
                            style={{
                                width: "100%", padding: "14px", borderRadius: 12, fontSize: 14,
                                fontWeight: 700, cursor: loading ? "not-allowed" : "pointer",
                                background: loading ? "rgba(52,211,153,0.15)" : "linear-gradient(135deg, #34d399, #10b981)",
                                border: "none", color: loading ? "#34d399" : "#fff",
                                display: "flex", alignItems: "center", justifyContent: "center", gap: 10,
                                transition: "all 0.2s",
                                boxShadow: loading ? "none" : "0 0 24px rgba(52,211,153,0.3)",
                            }}>
                            {loading
                                ? <><Loader2 size={16} className="animate-spin-sm" /> Analyzing with Random Forest…</>
                                : <><ClipboardList size={16} /> Run Clinical Assessment</>
                            }
                        </button>
                    </div>

                    {/* ── Results ── */}
                    {result && level && (
                        <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>

                            {/* Primary */}
                            <div style={{ background: level.bg, border: `1px solid ${level.border}`, borderRadius: 22, padding: 28 }}>
                                <div style={{ fontSize: 11, color: "#64748b", fontWeight: 700, letterSpacing: "0.07em", marginBottom: 10 }}>
                                    CLINICAL ASSESSMENT RESULT
                                </div>
                                <div style={{ fontSize: 30, fontWeight: 900, color: level.color, letterSpacing: "-0.03em", marginBottom: 6 }}>
                                    {result.class_name}
                                </div>
                                <div style={{ fontSize: 13, color: "#94a3b8", marginBottom: 20 }}>
                                    Risk Level: <strong style={{ color: level.color }}>{result.risk_level}</strong>
                                </div>

                                <div style={{
                                    display: "flex", alignItems: "center", justifyContent: "space-between",
                                    background: "rgba(255,255,255,0.06)", borderRadius: 12, padding: "14px 16px"
                                }}>
                                    <div style={{ textAlign: "center" }}>
                                        <div style={{ fontSize: 28, fontWeight: 900, color: level.color }}>{result.confidence.toFixed(1)}%</div>
                                        <div style={{ fontSize: 11, color: "#64748b" }}>Confidence</div>
                                    </div>
                                    <div style={{ textAlign: "center" }}>
                                        <div style={{ fontSize: 28, fontWeight: 900, color: "#94a3b8" }}>Grade {result.grade}</div>
                                        <div style={{ fontSize: 11, color: "#64748b" }}>ICDR Scale</div>
                                    </div>
                                    <div style={{ textAlign: "center" }}>
                                        <div style={{ fontSize: 13, fontWeight: 700, color: "#94a3b8" }}>{result.processing_time}s</div>
                                        <div style={{ fontSize: 11, color: "#64748b" }}>Inference</div>
                                    </div>
                                </div>
                            </div>

                            {/* Risk Factors */}
                            {result.risk_factors.length > 0 && (
                                <div className="glass" style={{ borderRadius: 20, padding: 22 }}>
                                    <h3 style={{ fontSize: 11, fontWeight: 700, color: "#64748b", marginBottom: 14, letterSpacing: "0.07em" }}>
                                        IDENTIFIED RISK FACTORS
                                    </h3>
                                    <div style={{ display: "flex", flexDirection: "column", gap: 9 }}>
                                        {result.risk_factors.map((rf, i) => (
                                            <div key={i} style={{
                                                display: "flex", alignItems: "center", gap: 10,
                                                padding: "9px 14px", background: "rgba(239,68,68,0.08)",
                                                border: "1px solid rgba(239,68,68,0.18)", borderRadius: 9
                                            }}>
                                                <AlertTriangle size={13} color="#f87171" />
                                                <span style={{ fontSize: 12, color: "#f87171" }}>{rf}</span>
                                            </div>
                                        ))}
                                    </div>
                                </div>
                            )}

                            {/* Probability bars */}
                            <div className="glass" style={{ borderRadius: 20, padding: 22 }}>
                                <h3 style={{ fontSize: 11, fontWeight: 700, color: "#64748b", marginBottom: 16, letterSpacing: "0.07em" }}>
                                    GRADE PROBABILITIES
                                </h3>
                                <div style={{ display: "flex", flexDirection: "column", gap: 13 }}>
                                    {probs.map(([name, pct], idx) => {
                                        const lv = SEVERITY[idx]; const isTop = idx === result.grade
                                        return (
                                            <div key={name}>
                                                <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 5 }}>
                                                    <span style={{ fontSize: 12, color: isTop ? lv.color : "#64748b", fontWeight: isTop ? 700 : 400 }}>{name}</span>
                                                    <span style={{ fontSize: 12, color: isTop ? lv.color : "#475569", fontWeight: isTop ? 700 : 400 }}>{pct.toFixed(1)}%</span>
                                                </div>
                                                <div style={{ height: 6, borderRadius: 100, background: "rgba(255,255,255,0.05)", overflow: "hidden" }}>
                                                    <div style={{
                                                        height: "100%", width: `${pct}%`, borderRadius: 100,
                                                        background: isTop ? lv.color : "rgba(255,255,255,0.1)",
                                                        transition: "width 0.9s cubic-bezier(0.4,0,0.2,1)"
                                                    }} />
                                                </div>
                                            </div>
                                        )
                                    })}
                                </div>
                            </div>

                            {/* Recommendation */}
                            <div className="glass" style={{ borderRadius: 20, padding: 22, display: "flex", gap: 14 }}>
                                <div style={{
                                    width: 38, height: 38, borderRadius: 11,
                                    background: "rgba(52,211,153,0.12)", display: "flex", alignItems: "center",
                                    justifyContent: "center", flexShrink: 0
                                }}>
                                    <Info size={17} color="#34d399" />
                                </div>
                                <div>
                                    <div style={{ fontSize: 13, fontWeight: 700, color: "#e2e8f0", marginBottom: 7 }}>
                                        Clinical Recommendation
                                    </div>
                                    <div style={{ fontSize: 13, color: "#64748b", lineHeight: 1.75 }}>
                                        {result.recommendation}
                                    </div>
                                    <div style={{ fontSize: 11, color: "#475569", marginTop: 8 }}>
                                        Model: {result.model}
                                    </div>
                                </div>
                            </div>
                        </div>
                    )}
                </div>
            </main>

            <footer style={{ borderTop: "1px solid rgba(255,255,255,0.06)", padding: "24px", textAlign: "center" }}>
                <p style={{ fontSize: 12, color: "#334155" }}>
                    ⚠️ For screening assistance only. Always consult a licensed ophthalmologist.
                </p>
            </footer>
        </div>
    )
}
