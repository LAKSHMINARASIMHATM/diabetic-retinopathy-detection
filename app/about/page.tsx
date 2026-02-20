"use client"

import {
    Brain, BarChart3, Shield, Eye, CheckCircle, Activity,
    Database, Layers, Zap, GitBranch, FlaskConical
} from "lucide-react"
import { Navbar } from "@/components/Navbar"

const PIPELINE_STEPS = [
    { step: "01", icon: Eye, color: "#818cf8", title: "Image Acquisition", desc: "Fundus photograph captured via non-mydriatic camera. Supports JPEG, PNG, TIFF formats up to 20MB." },
    { step: "02", icon: FlaskConical, color: "#34d399", title: "Clinical Preprocessing", desc: "CLAHE contrast enhancement, circular crop to remove borders, BGR→RGB conversion, ImageNet normalization." },
    { step: "03", icon: Layers, color: "#f59e0b", title: "EfficientNetB3 Inference", desc: "Transfer-learned on ImageNet, fine-tuned in two stages on 250k+ labeled fundus images with class-weighted loss." },
    { step: "04", icon: Zap, color: "#f87171", title: "Test-Time Augmentation", desc: "3 TTA rounds average predictions over horizontal flips, rotations, and brightness shifts for robustness." },
    { step: "05", icon: Brain, color: "#c084fc", title: "Grad-CAM Visualization", desc: "Gradient-weighted class activation maps highlight retinal lesions — microaneurysms, hemorrhages, exudates." },
    { step: "06", icon: CheckCircle, color: "#22c55e", title: "ICDR Grade Output", desc: "5-level severity grade (0–4) with per-class confidence, risk level, and clinical recommendation." },
]

const MODEL_CARDS = [
    {
        color: "#818cf8",
        icon: Brain,
        title: "EfficientNetB3",
        badge: "Deep Learning · Image",
        rows: [
            ["Architecture", "EfficientNetB3 (Keras/TensorFlow)"],
            ["Pre-training", "ImageNet (1.28M images)"],
            ["Fine-tuning", "APTOS 2019 / EyePACS compatible"],
            ["Input Size", "224 × 224 × 3"],
            ["Output Classes", "5 (No DR → Proliferative DR)"],
            ["Training", "Two-stage: head-only → full fine-tune"],
            ["Regularization", "Dropout(0.4), L2(1e-4), CLAHE augment"],
            ["Augmentation", "Flips, rotate, brightness, Gaussian noise"],
        ],
    },
    {
        color: "#34d399",
        icon: BarChart3,
        title: "Random Forest",
        badge: "Ensemble · Clinical Features",
        rows: [
            ["Algorithm", "Random Forest Classifier (sklearn)"],
            ["Features", "24 engineered clinical features"],
            ["Training Set", "240 patients (80% split)"],
            ["Test Set", "60 patients (20% split)"],
            ["Estimators", "500 trees, max_depth=12"],
            ["Class Balance", "class_weight='balanced'"],
            ["Cross-Validation", "5-fold stratified CV"],
            ["Feature Eng.", "Interaction terms, VA encoding, MAP"],
        ],
    },
]

const FEATURES_TABLE = [
    { group: "Numeric", items: ["Age", "Diabetes Duration", "HbA1c %", "Systolic BP", "Diastolic BP", "BMI", "Cholesterol mg/dL", "IOP mmHg"] },
    { group: "Ophthalmic", items: ["VA Right Score", "VA Left Score", "VA Avg Score", "VA Diff", "Eye Side"] },
    { group: "Derived", items: ["MAP (Mean Arterial Pressure)", "Pulse Pressure", "HbA1c × Duration", "Age / (Duration + 1)"] },
    { group: "Categorical", items: ["Gender", "Ethnicity", "Diabetes Type", "Smoking", "Kidney Disease", "Neuropathy", "Has Symptom"] },
]

export default function AboutPage() {
    return (
        <div style={{ minHeight: "100vh", color: "#e2e8f0", fontFamily: "'Inter', sans-serif" }}>
            <Navbar />

            <main style={{ maxWidth: 1200, margin: "0 auto", padding: "48px 24px 80px" }}>

                {/* Header */}
                <div style={{ marginBottom: 60 }}>
                    <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 10 }}>
                        <div style={{
                            width: 36, height: 36, borderRadius: 10,
                            background: "rgba(168,139,250,0.15)", border: "1px solid rgba(168,139,250,0.25)",
                            display: "flex", alignItems: "center", justifyContent: "center"
                        }}>
                            <FlaskConical size={18} color="#a78bfa" />
                        </div>
                        <h1 style={{ fontSize: 24, fontWeight: 800, color: "#f1f5f9", letterSpacing: "-0.02em" }}>
                            About the System
                        </h1>
                    </div>
                    <p style={{ fontSize: 14, color: "#475569", maxWidth: 650, lineHeight: 1.75, marginLeft: 46 }}>
                        RetinaAI is a dual-AI engine for diabetic retinopathy (DR) detection. It combines a deep learning
                        convolutional model for fundus images with a supervised ensemble model for clinical tabular data. Both
                        models are trained on real-world data and deployed via a FastAPI backend.
                    </p>
                </div>

                {/* Model architecture cards */}
                <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit,minmax(340px,1fr))", gap: 24, marginBottom: 60 }}>
                    {MODEL_CARDS.map(({ color, icon: Icon, title, badge, rows }) => (
                        <div key={title} className="glass" style={{ borderRadius: 22, padding: 30, position: "relative", overflow: "hidden" }}>
                            <div style={{
                                position: "absolute", top: -50, right: -50, width: 160, height: 160,
                                borderRadius: "50%", background: `${color}15`, filter: "blur(50px)", pointerEvents: "none"
                            }} />

                            <div style={{ display: "flex", alignItems: "center", gap: 14, marginBottom: 24 }}>
                                <div style={{
                                    width: 48, height: 48, borderRadius: 14, background: `${color}18`,
                                    border: `1px solid ${color}30`, display: "flex", alignItems: "center", justifyContent: "center"
                                }}>
                                    <Icon size={22} color={color} />
                                </div>
                                <div>
                                    <div style={{ fontSize: 18, fontWeight: 800, color: "#f1f5f9" }}>{title}</div>
                                    <div style={{ fontSize: 11, color, fontWeight: 700, marginTop: 2 }}>{badge}</div>
                                </div>
                            </div>

                            <div style={{ display: "flex", flexDirection: "column", gap: 0 }}>
                                {rows.map(([k, v], i) => (
                                    <div key={k} style={{
                                        display: "flex", justifyContent: "space-between", alignItems: "flex-start",
                                        padding: "10px 0", gap: 16,
                                        borderBottom: i < rows.length - 1 ? "1px solid rgba(255,255,255,0.05)" : "none",
                                    }}>
                                        <span style={{ fontSize: 12, color: "#64748b", fontWeight: 600, flexShrink: 0 }}>{k}</span>
                                        <span style={{ fontSize: 12, color: "#94a3b8", textAlign: "right" }}>{v}</span>
                                    </div>
                                ))}
                            </div>
                        </div>
                    ))}
                </div>

                {/* Image inference pipeline */}
                <div className="glass" style={{ borderRadius: 22, padding: 36, marginBottom: 60 }}>
                    <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 36 }}>
                        <GitBranch size={18} color="#818cf8" />
                        <h2 style={{ fontSize: 16, fontWeight: 700, color: "#f1f5f9" }}>Image Analysis Pipeline</h2>
                    </div>
                    <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit,minmax(220px,1fr))", gap: 20 }}>
                        {PIPELINE_STEPS.map(({ step, icon: Icon, color, title, desc }) => (
                            <div key={step} style={{ position: "relative" }}>
                                <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 12 }}>
                                    <div style={{
                                        width: 32, height: 32, borderRadius: 9, background: `${color}18`,
                                        border: `1px solid ${color}30`, display: "flex", alignItems: "center", justifyContent: "center", flexShrink: 0
                                    }}>
                                        <Icon size={15} color={color} />
                                    </div>
                                    <div style={{ fontSize: 10, fontWeight: 800, color, letterSpacing: "0.06em" }}>STEP {step}</div>
                                </div>
                                <div style={{ fontSize: 13, fontWeight: 700, color: "#e2e8f0", marginBottom: 6 }}>{title}</div>
                                <div style={{ fontSize: 12, color: "#64748b", lineHeight: 1.7 }}>{desc}</div>
                            </div>
                        ))}
                    </div>
                </div>

                {/* Clinical features table */}
                <div className="glass" style={{ borderRadius: 22, padding: 36, marginBottom: 60 }}>
                    <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 28 }}>
                        <Database size={18} color="#34d399" />
                        <h2 style={{ fontSize: 16, fontWeight: 700, color: "#f1f5f9" }}>24 Clinical Features</h2>
                    </div>
                    <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit,minmax(240px,1fr))", gap: 20 }}>
                        {FEATURES_TABLE.map(({ group, items }) => (
                            <div key={group}>
                                <div style={{ fontSize: 11, fontWeight: 700, color: "#34d399", marginBottom: 12, letterSpacing: "0.05em" }}>
                                    {group.toUpperCase()}
                                </div>
                                <div style={{ display: "flex", flexDirection: "column", gap: 7 }}>
                                    {items.map(item => (
                                        <div key={item} style={{ display: "flex", alignItems: "center", gap: 8, fontSize: 13, color: "#64748b" }}>
                                            <Zap size={10} color="#34d399" /> {item}
                                        </div>
                                    ))}
                                </div>
                            </div>
                        ))}
                    </div>
                </div>

                {/* DR Grading Reference */}
                <div className="glass" style={{ borderRadius: 22, padding: 36 }}>
                    <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 28 }}>
                        <Activity size={18} color="#f59e0b" />
                        <h2 style={{ fontSize: 16, fontWeight: 700, color: "#f1f5f9" }}>
                            International Clinical Diabetic Retinopathy Severity Scale
                        </h2>
                    </div>
                    <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
                        {[
                            { g: 0, color: "#22c55e", label: "No DR", features: "No abnormalities detectable.", action: "Routine annual screening." },
                            { g: 1, color: "#84cc16", label: "Mild NPDR", features: "Microaneurysms only.", action: "Re-examine in 12 months." },
                            { g: 2, color: "#f59e0b", label: "Moderate NPDR", features: "Microaneurysms + hard exudates, flame hemorrhages, cotton wool spots < severe threshold.", action: "Ophthalmology referral 3–6 months." },
                            { g: 3, color: "#ef4444", label: "Severe NPDR", features: ">20 hemorrhages / quadrant, venous beading ≥2 quadrants, IRMA ≥1 quadrant.", action: "Urgent ophthalmology within 1–3 months." },
                            { g: 4, color: "#dc2626", label: "Proliferative DR", features: "Neovascularization, vitreous / pre-retinal hemorrhage.", action: "Immediate referral. Laser / anti-VEGF required." },
                        ].map(({ g, color, label, features, action }) => (
                            <div key={g} style={{
                                display: "grid", gridTemplateColumns: "52px 1fr 1fr", gap: 16, alignItems: "start",
                                padding: "16px 0", borderBottom: g < 4 ? "1px solid rgba(255,255,255,0.05)" : "none"
                            }}>
                                <div style={{
                                    width: 44, height: 44, borderRadius: 12, background: `${color}18`,
                                    border: `1px solid ${color}30`, display: "flex", alignItems: "center",
                                    justifyContent: "center", fontSize: 16, fontWeight: 800, color
                                }}>
                                    {g}
                                </div>
                                <div>
                                    <div style={{ fontSize: 13, fontWeight: 700, color, marginBottom: 5 }}>{label}</div>
                                    <div style={{ fontSize: 12, color: "#475569", lineHeight: 1.65 }}>{features}</div>
                                </div>
                                <div style={{
                                    padding: "8px 14px", background: `${color}10`, border: `1px solid ${color}20`,
                                    borderRadius: 9, fontSize: 12, color: "#64748b", lineHeight: 1.65
                                }}>
                                    <Shield size={10} color={color} style={{ marginRight: 6, verticalAlign: "middle" }} />
                                    {action}
                                </div>
                            </div>
                        ))}
                    </div>
                </div>
            </main>

            <footer style={{ borderTop: "1px solid rgba(255,255,255,0.06)", padding: "24px", textAlign: "center" }}>
                <p style={{ fontSize: 12, color: "#334155" }}>
                    ⚠️ For screening assistance and educational purposes only. Always consult a licensed ophthalmologist.
                </p>
            </footer>
        </div>
    )
}
