"use client"

import { useState, useRef, useCallback } from "react"
import {
    Upload, Eye, Activity, AlertTriangle, CheckCircle, Info,
    ChevronRight, Brain, Zap, ImageIcon, RefreshCw
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

type AnalysisResult = {
    grade: number
    class_name: string
    confidence: number
    probabilities: Record<string, number>
    risk_level: string
    recommendation: string
    processing_time: number
    filename: string
    gradcam_b64: string | null
    model: string
    demo_mode: boolean
}

export default function AnalyzePage() {
    const [isDragging, setIsDragging] = useState(false)
    const [imagePreview, setImagePreview] = useState<string | null>(null)
    const [result, setResult] = useState<AnalysisResult | null>(null)
    const [isAnalyzing, setIsAnalyzing] = useState(false)
    const [error, setError] = useState<string | null>(null)
    const [showGradcam, setShowGradcam] = useState(false)
    const fileInputRef = useRef<HTMLInputElement>(null)

    const analyzeImage = async (file: File) => {
        setIsAnalyzing(true)
        setResult(null)
        setError(null)
        setShowGradcam(false)

        const reader = new FileReader()
        reader.onload = (e) => setImagePreview(e.target?.result as string)
        reader.readAsDataURL(file)

        try {
            const formData = new FormData()
            formData.append("file", file)
            const res = await fetch(`${API_BASE}/predict`, {
                method: "POST",
                body: formData,
                signal: AbortSignal.timeout(60000),
            })
            if (!res.ok) {
                const err = await res.json().catch(() => ({ detail: "Unknown error" }))
                throw new Error(err.detail || `HTTP ${res.status}`)
            }
            const data: AnalysisResult = await res.json()
            setResult(data)
        } catch (e: any) {
            if (e.name === "TimeoutError" || e.name === "TypeError") {
                setError("Cannot reach the backend API. Make sure the FastAPI server is running on port 8000.")
            } else {
                setError(e.message || "Analysis failed. Please try again.")
            }
        } finally {
            setIsAnalyzing(false)
        }
    }

    const handleFile = (file: File) => {
        if (!file.type.startsWith("image/")) {
            setError("Please upload a valid retinal image (JPEG, PNG, TIFF).")
            return
        }
        if (file.size > 20 * 1024 * 1024) {
            setError("File size must be under 20MB.")
            return
        }
        analyzeImage(file)
    }

    const onDrop = useCallback((e: React.DragEvent) => {
        e.preventDefault(); setIsDragging(false)
        const file = e.dataTransfer.files[0]
        if (file) handleFile(file)
    }, [])

    const reset = () => {
        setImagePreview(null); setResult(null)
        setError(null); setShowGradcam(false)
        if (fileInputRef.current) fileInputRef.current.value = ""
    }

    const level = result !== null ? SEVERITY[result.grade] : null
    const probs = result ? Object.entries(result.probabilities) : []

    return (
        <div style={{ minHeight: "100vh", color: "#e2e8f0", fontFamily: "'Inter', sans-serif" }}>
            <Navbar />

            <main style={{ maxWidth: 1200, margin: "0 auto", padding: "48px 24px 80px" }}>
                {/* Page header */}
                <div style={{ marginBottom: 40 }}>
                    <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 10 }}>
                        <div style={{
                            width: 36, height: 36, borderRadius: 10,
                            background: "rgba(129,140,248,0.15)", border: "1px solid rgba(129,140,248,0.25)",
                            display: "flex", alignItems: "center", justifyContent: "center",
                        }}>
                            <Eye size={18} color="#818cf8" />
                        </div>
                        <h1 style={{ fontSize: 24, fontWeight: 800, color: "#f1f5f9", letterSpacing: "-0.02em" }}>
                            Fundus Image Analysis
                        </h1>
                    </div>
                    <p style={{ fontSize: 14, color: "#475569", maxWidth: 580, lineHeight: 1.7, marginLeft: 46 }}>
                        Upload a retinal fundus photograph. The EfficientNetB3 model (trained on 250k+ images) will classify DR severity with a Grad-CAM attention heatmap.
                    </p>
                </div>

                <div style={{
                    display: "grid",
                    gridTemplateColumns: result ? "1fr 1fr" : "minmax(400px,600px)",
                    gap: 28, justifyContent: "center", alignItems: "start",
                }}>
                    {/* ── Upload Panel ── */}
                    <div className="glass" style={{ borderRadius: 22, padding: 28 }}>
                        <h2 style={{
                            fontSize: 14, fontWeight: 700, color: "#94a3b8", marginBottom: 20,
                            display: "flex", alignItems: "center", gap: 8, letterSpacing: "0.04em"
                        }}>
                            <Upload size={14} color="#818cf8" /> UPLOAD RETINAL IMAGE
                        </h2>

                        {/* Drop zone */}
                        <div
                            onClick={() => fileInputRef.current?.click()}
                            onDrop={onDrop}
                            onDragOver={(e) => { e.preventDefault(); setIsDragging(true) }}
                            onDragLeave={() => setIsDragging(false)}
                            style={{
                                border: `2px dashed ${isDragging ? "#818cf8" : "rgba(129,140,248,0.22)"}`,
                                borderRadius: 16, padding: "40px 24px", textAlign: "center",
                                cursor: "pointer",
                                background: isDragging ? "rgba(129,140,248,0.09)" : "rgba(129,140,248,0.03)",
                                transition: "all 0.25s", minHeight: 240,
                                display: "flex", flexDirection: "column", alignItems: "center",
                                justifyContent: "center", gap: 14,
                            }}
                        >
                            {imagePreview ? (
                                <img src={imagePreview} alt="Preview"
                                    style={{ maxHeight: 260, maxWidth: "100%", borderRadius: 12, objectFit: "cover" }} />
                            ) : (
                                <>
                                    <div style={{
                                        width: 60, height: 60, borderRadius: 18,
                                        background: "rgba(129,140,248,0.12)",
                                        display: "flex", alignItems: "center", justifyContent: "center",
                                    }}>
                                        <ImageIcon size={28} color="#818cf8" />
                                    </div>
                                    <div>
                                        <div style={{ fontWeight: 700, color: "#e2e8f0", fontSize: 15 }}>Drop retinal fundus image</div>
                                        <div style={{ color: "#64748b", fontSize: 13, marginTop: 4 }}>or click to browse files</div>
                                    </div>
                                    <div style={{ fontSize: 11, color: "#475569", background: "rgba(255,255,255,0.04)", padding: "6px 14px", borderRadius: 100 }}>
                                        JPEG · PNG · TIFF · BMP · up to 20 MB
                                    </div>
                                </>
                            )}
                            <input ref={fileInputRef} type="file" accept="image/*" onChange={(e) => e.target.files?.[0] && handleFile(e.target.files[0])} style={{ display: "none" }} />
                        </div>

                        {/* Error */}
                        {error && (
                            <div style={{
                                marginTop: 16, padding: "12px 16px", background: "rgba(239,68,68,0.09)",
                                border: "1px solid rgba(239,68,68,0.25)", borderRadius: 10, display: "flex", gap: 10, alignItems: "center"
                            }}>
                                <AlertTriangle size={15} color="#f87171" />
                                <span style={{ fontSize: 13, color: "#f87171" }}>{error}</span>
                            </div>
                        )}

                        {/* Analyzing spinner */}
                        {isAnalyzing && (
                            <div style={{ marginTop: 24, textAlign: "center" }}>
                                <div style={{
                                    display: "inline-flex", alignItems: "center", gap: 10,
                                    background: "rgba(129,140,248,0.1)", border: "1px solid rgba(129,140,248,0.2)",
                                    borderRadius: 100, padding: "12px 22px"
                                }}>
                                    <div style={{
                                        width: 14, height: 14, borderRadius: "50%",
                                        border: "2px solid #818cf8", borderTopColor: "transparent"
                                    }}
                                        className="animate-spin-sm" />
                                    <span style={{ fontSize: 13, color: "#818cf8", fontWeight: 600 }}>
                                        Analyzing with EfficientNetB3…
                                    </span>
                                </div>
                                <div style={{ marginTop: 10, height: 3, borderRadius: 100 }} className="loading-bar" />
                            </div>
                        )}

                        {/* Action buttons */}
                        {imagePreview && !isAnalyzing && (
                            <div style={{ display: "flex", gap: 10, marginTop: 16 }}>
                                <button onClick={() => fileInputRef.current?.click()}
                                    style={{
                                        flex: 1, padding: "10px", borderRadius: 10,
                                        background: "rgba(129,140,248,0.1)", border: "1px solid rgba(129,140,248,0.25)",
                                        color: "#818cf8", fontSize: 13, fontWeight: 600, cursor: "pointer"
                                    }}>
                                    Upload New Image
                                </button>
                                <button onClick={reset}
                                    style={{
                                        padding: "10px 14px", borderRadius: 10,
                                        background: "rgba(255,255,255,0.04)", border: "1px solid rgba(255,255,255,0.08)",
                                        color: "#64748b", fontSize: 13, cursor: "pointer"
                                    }}>
                                    <RefreshCw size={14} />
                                </button>
                            </div>
                        )}

                        {/* Instructions */}
                        <div style={{
                            marginTop: 22, padding: "14px 16px", background: "rgba(129,140,248,0.05)",
                            border: "1px solid rgba(129,140,248,0.12)", borderRadius: 12
                        }}>
                            <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 8 }}>
                                <Zap size={13} color="#818cf8" />
                                <span style={{ fontSize: 12, fontWeight: 700, color: "#818cf8" }}>Start the AI Backend</span>
                            </div>
                            <code style={{
                                display: "block", fontSize: 11, color: "#64748b", background: "rgba(0,0,0,0.3)",
                                borderRadius: 7, padding: "8px 12px"
                            }}>
                                cd backend<br />
                                uvicorn server:app --reload --port 8000
                            </code>
                        </div>
                    </div>

                    {/* ── Results Panel ── */}
                    {result && level && (
                        <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>

                            {/* Primary Result Card */}
                            <div style={{ background: level.bg, border: `1px solid ${level.border}`, borderRadius: 22, padding: 28 }}>
                                <div style={{ display: "flex", alignItems: "flex-start", justifyContent: "space-between", marginBottom: 20 }}>
                                    <div>
                                        <div style={{ fontSize: 11, color: "#64748b", fontWeight: 700, letterSpacing: "0.07em", marginBottom: 8 }}>
                                            DIAGNOSIS RESULT
                                        </div>
                                        <div style={{ fontSize: 30, fontWeight: 900, color: level.color, letterSpacing: "-0.03em" }}>
                                            {result.class_name}
                                        </div>
                                        <div style={{ fontSize: 13, color: "#94a3b8", marginTop: 5 }}>
                                            Risk Level: <strong style={{ color: level.color }}>{result.risk_level}</strong>
                                        </div>
                                    </div>
                                    <div style={{ textAlign: "right" }}>
                                        <div style={{ fontSize: 42, fontWeight: 900, color: level.color, letterSpacing: "-0.04em", lineHeight: 1 }}>
                                            {result.confidence.toFixed(1)}%
                                        </div>
                                        <div style={{ fontSize: 11, color: "#64748b", marginTop: 3 }}>Confidence</div>
                                        <div style={{ fontSize: 11, color: "#475569", marginTop: 2 }}>
                                            ⏱ {result.processing_time}s
                                        </div>
                                    </div>
                                </div>

                                <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
                                    <div style={{
                                        display: "flex", alignItems: "center", gap: 6,
                                        background: "rgba(255,255,255,0.06)", borderRadius: 8, padding: "5px 12px"
                                    }}>
                                        <Activity size={12} color="#64748b" />
                                        <span style={{ fontSize: 12, color: "#94a3b8" }}>Grade {result.grade}</span>
                                    </div>
                                    <div style={{
                                        display: "flex", alignItems: "center", gap: 6,
                                        background: "rgba(255,255,255,0.06)", borderRadius: 8, padding: "5px 12px"
                                    }}>
                                        <Brain size={12} color="#818cf8" />
                                        <span style={{ fontSize: 12, color: "#94a3b8" }}>{result.model}</span>
                                    </div>
                                    {result.grade === 0 && (
                                        <div style={{
                                            display: "flex", alignItems: "center", gap: 6,
                                            background: "rgba(34,197,94,0.1)", borderRadius: 8, padding: "5px 12px"
                                        }}>
                                            <CheckCircle size={12} color="#22c55e" />
                                            <span style={{ fontSize: 12, color: "#22c55e" }}>Healthy Retina</span>
                                        </div>
                                    )}
                                    {result.grade >= 3 && (
                                        <div style={{
                                            display: "flex", alignItems: "center", gap: 6,
                                            background: "rgba(239,68,68,0.1)", borderRadius: 8, padding: "5px 12px"
                                        }}>
                                            <AlertTriangle size={12} color="#ef4444" />
                                            <span style={{ fontSize: 12, color: "#ef4444" }}>Urgent Referral</span>
                                        </div>
                                    )}
                                </div>
                            </div>

                            {/* Probability Bars */}
                            <div className="glass" style={{ borderRadius: 20, padding: 24 }}>
                                <h3 style={{ fontSize: 11, fontWeight: 700, color: "#64748b", marginBottom: 18, letterSpacing: "0.07em" }}>
                                    GRADE PROBABILITIES
                                </h3>
                                <div style={{ display: "flex", flexDirection: "column", gap: 13 }}>
                                    {probs.map(([name, pct], idx) => {
                                        const lv = SEVERITY[idx]
                                        const isTop = idx === result.grade
                                        return (
                                            <div key={name}>
                                                <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 5 }}>
                                                    <span style={{ fontSize: 12, color: isTop ? lv.color : "#64748b", fontWeight: isTop ? 700 : 400 }}>{name}</span>
                                                    <span style={{ fontSize: 12, color: isTop ? lv.color : "#475569", fontWeight: isTop ? 700 : 400 }}>{pct.toFixed(1)}%</span>
                                                </div>
                                                <div style={{ height: 7, borderRadius: 100, background: "rgba(255,255,255,0.05)", overflow: "hidden" }}>
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

                            {/* Grad-CAM */}
                            {result.gradcam_b64 && (
                                <div className="glass" style={{ borderRadius: 20, padding: 20 }}>
                                    <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 14 }}>
                                        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                                            <Brain size={14} color="#818cf8" />
                                            <h3 style={{ fontSize: 12, fontWeight: 700, color: "#94a3b8", letterSpacing: "0.06em" }}>
                                                GRAD-CAM ATTENTION MAP
                                            </h3>
                                        </div>
                                        <button onClick={() => setShowGradcam(!showGradcam)}
                                            style={{
                                                fontSize: 12, color: "#818cf8", background: "rgba(129,140,248,0.1)",
                                                border: "1px solid rgba(129,140,248,0.2)", borderRadius: 7,
                                                padding: "4px 10px", cursor: "pointer", fontWeight: 600
                                            }}>
                                            {showGradcam ? "Hide" : "Show"}
                                        </button>
                                    </div>
                                    {showGradcam && (
                                        <div>
                                            <img
                                                src={`data:image/png;base64,${result.gradcam_b64}`}
                                                alt="Grad-CAM heatmap"
                                                style={{ width: "100%", borderRadius: 12, maxHeight: 280, objectFit: "cover" }}
                                            />
                                            <p style={{ fontSize: 11, color: "#475569", marginTop: 10, lineHeight: 1.6 }}>
                                                🔴 Red/warm areas indicate regions the model focused on for the prediction.
                                                Helps identify retinal lesions and microaneurysms driving the DR grade.
                                            </p>
                                        </div>
                                    )}
                                </div>
                            )}

                            {/* Recommendation */}
                            <div className="glass" style={{ borderRadius: 20, padding: 22, display: "flex", gap: 16, alignItems: "flex-start" }}>
                                <div style={{
                                    width: 38, height: 38, borderRadius: 11,
                                    background: "rgba(129,140,248,0.12)", display: "flex", alignItems: "center",
                                    justifyContent: "center", flexShrink: 0
                                }}>
                                    <Info size={18} color="#818cf8" />
                                </div>
                                <div>
                                    <div style={{ fontSize: 13, fontWeight: 700, color: "#e2e8f0", marginBottom: 7 }}>
                                        Clinical Recommendation
                                    </div>
                                    <div style={{ fontSize: 13, color: "#64748b", lineHeight: 1.75 }}>{result.recommendation}</div>
                                    <div style={{ fontSize: 11, color: "#475569", marginTop: 10, display: "flex", alignItems: "center", gap: 5 }}>
                                        <Info size={10} /> Not a substitute for professional ophthalmologic evaluation.
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
