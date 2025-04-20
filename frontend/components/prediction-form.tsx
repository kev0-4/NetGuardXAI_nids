"use client"

import type React from "react"

import { useState } from "react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Loader2 } from "lucide-react"

interface PredictionFormProps {
  onSubmit: (data: any) => void
  isLoading: boolean
}

// Define the input ranges
const inputRanges = {
  "Avg Packet Size": { min: 50.0, max: 200.0 },
  "Packet Length Mean": { min: 50.0, max: 150.0 },
  "Bwd Packet Length Std": { min: 0.0, max: 100.0 },
  "Packet Length Variance": { min: 0.0, max: 5000.0 },
  "Bwd Packet Length Max": { min: 0.0, max: 500.0 },
  "Packet Length Max": { min: 100.0, max: 1000.0 },
  "Packet Length Std": { min: 0.0, max: 200.0 },
  "Fwd Packet Length Mean": { min: 50.0, max: 150.0 },
  "Avg Fwd Segment Size": { min: 50.0, max: 150.0 },
  "Flow Bytes/s": { min: 1000.0, max: 20000.0 },
  "Avg Bwd Segment Size": { min: 50.0, max: 150.0 },
  "Bwd Packet Length Mean": { min: 50.0, max: 150.0 },
  "Fwd Packets/s": { min: 10.0, max: 100.0 },
  "Flow Packets/s": { min: 10.0, max: 200.0 },
  "Init Fwd Win Bytes": { min: 1024.0, max: 65535.0 },
  "Subflow Fwd Bytes": { min: 100.0, max: 1000.0 },
  "Fwd Packets Length Total": { min: 100.0, max: 2000.0 },
  "Fwd Act Data Packets": { min: 1.0, max: 20.0 },
  "Total Fwd Packets": { min: 1.0, max: 50.0 },
  "Subflow Fwd Packets": { min: 1.0, max: 50.0 },
}

export default function PredictionForm({ onSubmit, isLoading }: PredictionFormProps) {
  const [formData, setFormData] = useState({
    "Avg Packet Size": 120.0,
    "Packet Length Mean": 100.0,
    "Bwd Packet Length Std": 55.0,
    "Packet Length Variance": 3000.0,
    "Bwd Packet Length Max": 250.0,
    "Packet Length Max": 350.0,
    "Packet Length Std": 65.0,
    "Fwd Packet Length Mean": 85.0,
    "Avg Fwd Segment Size": 85.0,
    "Flow Bytes/s": 12000.0,
    "Avg Bwd Segment Size": 95.0,
    "Bwd Packet Length Mean": 90.0,
    "Fwd Packets/s": 60.0,
    "Flow Packets/s": 110.0,
    "Init Fwd Win Bytes": 8192.0,
    "Subflow Fwd Bytes": 600.0,
    "Fwd Packets Length Total": 700.0,
    "Fwd Act Data Packets": 6.0,
    "Total Fwd Packets": 12.0,
    "Subflow Fwd Packets": 12.0,
  })

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const { name, value } = e.target
    setFormData((prev) => ({
      ...prev,
      [name]: Number.parseFloat(value),
    }))
  }

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    onSubmit(formData)
  }

  const handleReset = () => {
    setFormData({
      "Avg Packet Size": 120.0,
      "Packet Length Mean": 100.0,
      "Bwd Packet Length Std": 55.0,
      "Packet Length Variance": 3000.0,
      "Bwd Packet Length Max": 250.0,
      "Packet Length Max": 350.0,
      "Packet Length Std": 65.0,
      "Fwd Packet Length Mean": 85.0,
      "Avg Fwd Segment Size": 85.0,
      "Flow Bytes/s": 12000.0,
      "Avg Bwd Segment Size": 95.0,
      "Bwd Packet Length Mean": 90.0,
      "Fwd Packets/s": 60.0,
      "Flow Packets/s": 110.0,
      "Init Fwd Win Bytes": 8192.0,
      "Subflow Fwd Bytes": 600.0,
      "Fwd Packets Length Total": 700.0,
      "Fwd Act Data Packets": 6.0,
      "Total Fwd Packets": 12.0,
      "Subflow Fwd Packets": 12.0,
    })
  }

  return (
    <form onSubmit={handleSubmit} className="space-y-6">
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {Object.entries(formData).map(([key, value]) => (
          <div key={key} className="space-y-2">
            <Label htmlFor={key} className="text-white">
              {key}
            </Label>
            <div className="space-y-1">
              <Input
                id={key}
                name={key}
                type="number"
                step="0.1"
                value={value}
                onChange={handleChange}
                min={inputRanges[key as keyof typeof inputRanges].min}
                max={inputRanges[key as keyof typeof inputRanges].max}
                className="bg-gray-700/50 border-gray-600 text-white"
              />
              <p className="text-xs text-gray-400">
                Range: {inputRanges[key as keyof typeof inputRanges].min} -{" "}
                {inputRanges[key as keyof typeof inputRanges].max}
              </p>
            </div>
          </div>
        ))}
      </div>

      <div className="flex gap-4">
        <Button type="submit" className="bg-emerald-500 hover:bg-emerald-600 text-white" disabled={isLoading}>
          {isLoading && <Loader2 className="mr-2 h-4 w-4 animate-spin" />}
          {isLoading ? "Analyzing..." : "Analyze Traffic"}
        </Button>
        <Button
          type="button"
          variant="outline"
          className="border-gray-600 text-white hover:bg-gray-700"
          onClick={handleReset}
          disabled={isLoading}
        >
          Reset to Default
        </Button>
      </div>
    </form>
  )
}
