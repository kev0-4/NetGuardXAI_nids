"use client"

import { useState } from "react"
import { ArrowLeft, Shield, AlertCircle, CheckCircle2, Loader2 } from "lucide-react"
import Link from "next/link"
import Image from "next/image"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert"
import PredictionForm from "@/components/prediction-form"
import MarkdownRenderer from "@/components/markdown-renderer"
import { predictNetworkTraffic } from "@/lib/api"

export default function PredictionPage() {
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [result, setResult] = useState<any>(null)
  const [activeTab, setActiveTab] = useState("summary")
  const [jsonInput, setJsonInput] = useState<string>(`{
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
    "Subflow Fwd Packets": 12.0
}`)

  const handleSubmit = async (formData: any) => {
    setLoading(true)
    setError(null)

    try {
      const data = await predictNetworkTraffic(formData)
      setResult(data)
    } catch (err) {
      setError("Failed to get prediction. Please check if the API server is running at http://localhost:5000")
      console.error(err)
    } finally {
      setLoading(false)
    }
  }

  const handleJsonSubmit = () => {
    try {
      const parsedData = JSON.parse(jsonInput)
      handleSubmit(parsedData)
    } catch (err) {
      setError("Invalid JSON format. Please check your input.")
    }
  }

  return (
    <div className="min-h-screen bg-gradient-to-b from-gray-900 to-gray-950">
      <header className="container mx-auto py-6 px-4 flex justify-between items-center">
        <Link href="/" className="flex items-center gap-2">
          <Shield className="h-8 w-8 text-emerald-500" />
          <span className="text-xl font-bold text-white">NetGuardXAI</span>
        </Link>
        <Link href="/">
          <Button variant="ghost" className="text-gray-300 hover:text-white">
            <ArrowLeft className="mr-2 h-4 w-4" /> Back to Home
          </Button>
        </Link>
      </header>

      <main className="container mx-auto py-10 px-4">
        <div className="max-w-7xl mx-auto">
          <h1 className="text-3xl md:text-4xl font-bold text-white mb-6">Network Intrusion Detection Demo</h1>
          <p className="text-xl text-gray-300 mb-10 max-w-3xl">
            Enter network traffic parameters below to analyze potential intrusions with our explainable AI model.
          </p>

          <div className="grid lg:grid-cols-2 gap-8">
            <div>
              <Card className="bg-gray-800/50 border-gray-700">
                <CardHeader>
                  <CardTitle className="text-white">Input Parameters</CardTitle>
                  <CardDescription>Enter network traffic parameters to analyze</CardDescription>
                </CardHeader>
                <CardContent>
                  <Tabs defaultValue="form" className="w-full">
                    <TabsList className="bg-gray-700/50 mb-6">
                      <TabsTrigger value="form">Form Input</TabsTrigger>
                      <TabsTrigger value="json">JSON Input</TabsTrigger>
                    </TabsList>
                    <TabsContent value="form">
                      <PredictionForm onSubmit={handleSubmit} isLoading={loading} />
                    </TabsContent>
                    <TabsContent value="json">
                      <div className="space-y-4">
                        <div className="relative">
                          <textarea
                            className="w-full h-96 p-4 bg-gray-700/30 text-gray-300 rounded-lg border border-gray-600 font-mono text-sm focus:outline-none focus:ring-2 focus:ring-emerald-500 focus:border-transparent"
                            value={jsonInput}
                            onChange={(e) => setJsonInput(e.target.value)}
                          />
                        </div>
                        <Button
                          onClick={handleJsonSubmit}
                          disabled={loading}
                          className="w-full bg-emerald-600 hover:bg-emerald-700 text-white"
                        >
                          {loading ? (
                            <>
                              <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                              Analyzing...
                            </>
                          ) : (
                            "Analyze JSON Data"
                          )}
                        </Button>
                      </div>
                    </TabsContent>
                  </Tabs>
                </CardContent>
              </Card>
            </div>

            <div>
              {loading && (
                <div className="flex flex-col items-center justify-center h-full">
                  <Loader2 className="h-12 w-12 text-emerald-500 animate-spin mb-4" />
                  <p className="text-white text-lg">Analyzing network traffic...</p>
                </div>
              )}

              {error && (
                <Alert variant="destructive" className="bg-red-900/50 border-red-800 text-white">
                  <AlertCircle className="h-4 w-4" />
                  <AlertTitle>Error</AlertTitle>
                  <AlertDescription>{error}</AlertDescription>
                </Alert>
              )}

              {!loading && !error && result && (
                <Card className="bg-gray-800/50 border-gray-700">
                  <CardHeader>
                    <div className="flex items-center justify-between">
                      <CardTitle className="text-white">
                        Prediction:{" "}
                        <span className={result.lime.prediction === "Benign" ? "text-emerald-500" : "text-red-500"}>
                          {result.lime.prediction}
                        </span>
                      </CardTitle>
                      <div className="flex items-center gap-2">
                        {result.lime.prediction === "Benign" ? (
                          <CheckCircle2 className="h-5 w-5 text-emerald-500" />
                        ) : (
                          <AlertCircle className="h-5 w-5 text-red-500" />
                        )}
                        <span className="text-white font-medium">
                          {(result.lime.confidence * 100).toFixed(2)}% confidence
                        </span>
                      </div>
                    </div>
                    <CardDescription>Explainable AI analysis of network traffic</CardDescription>
                  </CardHeader>
                  <CardContent>
                    <Tabs defaultValue="summary" value={activeTab} onValueChange={setActiveTab}>
                      <TabsList className="bg-gray-700/50 mb-6">
                        <TabsTrigger value="summary">Summary</TabsTrigger>
                        <TabsTrigger value="lime">LIME Explanation</TabsTrigger>
                        <TabsTrigger value="integrated">Integrated Gradients</TabsTrigger>
                      </TabsList>

                      <TabsContent value="summary" className="space-y-4">
                        <div className="bg-gray-700/30 p-4 rounded-lg border border-gray-700">
                          <h3 className="text-lg font-medium text-white mb-2">AI Summary</h3>
                          <MarkdownRenderer content={result.gemini_summary} />
                        </div>
                      </TabsContent>

                      <TabsContent value="lime" className="space-y-4">
                        <div className="bg-gray-700/30 p-4 rounded-lg border border-gray-700">
                          <h3 className="text-lg font-medium text-white mb-2">LIME Explanation</h3>
                          <ul className="space-y-2">
                            {result.lime.explanation.map((item: string, index: number) => (
                              <li key={index} className="text-gray-300">
                                {item}
                              </li>
                            ))}
                          </ul>
                        </div>

                        {result.lime.plot_path && (
                          <div className="mt-4">
                            <h3 className="text-lg font-medium text-white mb-2">Feature Importance Visualization</h3>
                            <div className="bg-white p-2 rounded-lg">
                              <Image
                                src={`http://localhost:5000/plots/${result.lime.plot_path.replace("plots\\", "")}`}
                                alt="LIME Explanation"
                                width={600}
                                height={400}
                                className="w-full h-auto"
                              />
                            </div>
                          </div>
                        )}
                      </TabsContent>

                      <TabsContent value="integrated" className="space-y-4">
                        <div className="bg-gray-700/30 p-4 rounded-lg border border-gray-700">
                          <h3 className="text-lg font-medium text-white mb-2">Integrated Gradients</h3>
                          <p className="text-gray-300">
                            Prediction:{" "}
                            <span
                              className={
                                result.integrated_gradients.prediction === "Benign"
                                  ? "text-emerald-500"
                                  : "text-red-500"
                              }
                            >
                              {result.integrated_gradients.prediction}
                            </span>{" "}
                            with {(result.integrated_gradients.confidence * 100).toFixed(2)}% confidence
                          </p>
                        </div>

                        {result.integrated_gradients.bar_plot_path && (
                          <div className="mt-4">
                            <h3 className="text-lg font-medium text-white mb-2">Feature Importance Bar Plot</h3>
                            <div className="bg-white p-2 rounded-lg">
                              <Image
                                src={`http://localhost:5000/plots/${result.integrated_gradients.bar_plot_path.replace("plots\\", "")}`}
                                alt="Integrated Gradients Bar Plot"
                                width={600}
                                height={400}
                                className="w-full h-auto"
                              />
                            </div>
                          </div>
                        )}

                        {result.integrated_gradients.heatmap_path && (
                          <div className="mt-4">
                            <h3 className="text-lg font-medium text-white mb-2">Feature Importance Heatmap</h3>
                            <div className="bg-white p-2 rounded-lg">
                              <Image
                                src={`http://localhost:5000/plots/${result.integrated_gradients.heatmap_path.replace("plots\\", "")}`}
                                alt="Integrated Gradients Heatmap"
                                width={600}
                                height={400}
                                className="w-full h-auto"
                              />
                            </div>
                          </div>
                        )}
                      </TabsContent>
                    </Tabs>
                  </CardContent>
                </Card>
              )}

              {!loading && !error && !result && (
                <div className="flex flex-col items-center justify-center h-full bg-gray-800/30 rounded-lg border border-gray-700 p-8">
                  <div className="bg-gray-700/30 p-4 rounded-full mb-4">
                    <Shield className="h-12 w-12 text-emerald-500" />
                  </div>
                  <h3 className="text-xl font-medium text-white mb-2">Enter Parameters to Start</h3>
                  <p className="text-gray-300 text-center max-w-md">
                    Fill in the network traffic parameters on the left and submit to see the AI analysis and
                    explanation.
                  </p>
                </div>
              )}
            </div>
          </div>
        </div>
      </main>

      <footer className="bg-gray-950 py-8 border-t border-gray-800 mt-20">
        <div className="container mx-auto px-4 text-center text-gray-400">
          <p>© {new Date().getFullYear()} NetGuardXAI. All rights reserved.</p>
        </div>
      </footer>
    </div>
  )
}