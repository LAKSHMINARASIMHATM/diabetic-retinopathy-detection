import type { Metadata } from 'next'
import { Inter } from 'next/font/google'
import './globals.css'

const inter = Inter({ subsets: ['latin'], variable: '--font-inter' })

export const metadata: Metadata = {
  title: 'RetinaAI — Diabetic Retinopathy Detection',
  description:
    'AI-powered diabetic retinopathy grading using EfficientNetB3 deep learning and clinical Random Forest models. 5-level DR severity classification from fundus images and clinical data.',
  keywords: ['diabetic retinopathy', 'DR detection', 'fundus analysis', 'EfficientNetB3', 'ophthalmology AI'],
}

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode
}>) {
  return (
    <html lang="en" className="dark">
      <body className={`${inter.variable} font-sans antialiased`}>
        {children}
      </body>
    </html>
  )
}
